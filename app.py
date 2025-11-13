"""
app.py

Police Recognition Analytics Platform
- Uses NER + gazetteer + fuzzy matching to extract officer names, departments, locations.
- Sentiment, summarization, QA, PDF summary export, CSV/JSON bulk export.
- Works with text input or uploaded TXT/PDF files.

Place datasets in the same folder:
 - OdishaIPCCrimedata.json
 - DistrictReport.json
 - mock_cctnsdata.json
 - publicfeedback.json

Run:
  pip install -r requirements.txt
  streamlit run app.py
"""

import streamlit as st
import json
import re
from datetime import datetime
from io import BytesIO
from typing import List, Dict, Optional, Tuple
import os
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

# NLP libs
from transformers import pipeline
import torch
import pdfplumber
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from rapidfuzz import process as rf_process
from rapidfuzz import fuzz
from dateutil import parser as dateparser

# -----------------------
# Configuration / files
# -----------------------
DATA_FILES = {
    "odisha_ipc": "OdishaIPCCrimedata.json",
    "district_reports": "DistrictReport.json",
    "mock_ctns": "mock_cctnsdata.json",
    "public_feedback": "publicfeedback.json"
}

# Officer DB fallback (if original OFFICER_DATABASE not present)
HARDCODED_OFFICERS = {
    "names": [
        "Officer John Smith", "Sergeant Mary Johnson", "Inspector David Brown",
        "Captain Sarah Williams", "Detective Michael Jones", "Officer Emily Davis",
        "Lieutenant Robert Miller", "Chief Patricia Wilson", "Officer James Moore",
        "Constable Jennifer Taylor", "Officer Christopher Anderson", "SI Rajesh Kumar",
        "ASI Priya Sharma", "Inspector Amit Patel", "Constable Sunita Verma", "Officer Rahul Singh",
        "Detective Anita Desai", "Captain Vijay Reddy", "SI Lakshmi Iyer", "Officer Arjun Nair",
        "Inspector Kavita Menon", "Constable Ravi Krishnan", "Officer Deepak Gupta", "Sergeant Pooja Rao"
    ],
    "departments": [
        "Central Police Station", "North District Police", "South Precinct",
        "East Division Police", "West Police Department", "Metropolitan Police",
        "City Police Commissionerate", "Traffic Police Department", "Crime Investigation Department",
        "Special Task Force", "Cyber Crime Unit", "14th Precinct", "5th District Station",
        "Bhubaneswar Police", "Cuttack Police", "Puri Police Station", "Kendrapara Police"
    ],
    "locations": [
        "Downtown", "Riverside", "Central District", "North Zone",
        "South Sector", "East Block", "West Avenue", "Main Street",
        "Park Avenue", "City Center", "Bangalore", "Mumbai", "Delhi",
        "Kolkata", "Chennai", "Bhubaneswar", "Cuttack", "Puri"
    ]
}

# Standard list of crime phrases (from your fields list) to match in English text
CRIME_KEYWORDS = [
    "murder", "rape", "kidnapping", "abduction", "dacoity", "robbery", "burglary",
    "theft", "riots", "cheating", "counterfeiting", "arson", "hurt", "dowry death",
    "assault", "insult to the modesty", "cruelty", "importation of girls",
    "causing death by negligence", "other ipc crimes"
]

TITLE_PATTERNS = r'\b(Officer|Constable|Inspector|Sergeant|Detective|Chief|Captain|Lieutenant|SI|ASI|Sub-Inspector|Sub Inspector|PSI|ASI|Inspector)\b\.?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'

# -----------------------
# Utility: load external JSON datasets if present
# -----------------------
def load_json_file_safe(path: str):
    if os.path.exists(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            st.warning(f"Could not load {path}: {e}")
            return None
    return None

def load_all_datasets():
    d = {}
    for k, p in DATA_FILES.items():
        d[k] = load_json_file_safe(p)
    return d

# -----------------------
# Models (cached)
# -----------------------
@st.cache_resource(show_spinner=False)
def load_models(device: Optional[int] = None):
    """
    Loads transformers pipelines:
     - NER (token-classification) - english model
     - sentiment
     - summarizer
     - QA
    """
    try:
        # choose device: GPU (0) if available else CPU (-1)
        if device is None:
            device_id = 0 if torch.cuda.is_available() else -1
        else:
            device_id = device

        # NER: English pretrained NER
        ner_model = pipeline(
            "token-classification",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple",
            device=device_id
        )

        sentiment = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device_id
        )

        summarizer = pipeline(
            "summarization",
            model="facebook/bart-large-cnn",
            device=device_id
        )

        qa = pipeline(
            "question-answering",
            model="deepset/roberta-base-squad2",
            device=device_id
        )

        return {
            "ner": ner_model,
            "sentiment": sentiment,
            "summarizer": summarizer,
            "qa": qa
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# -----------------------
# Extraction helpers
# -----------------------
def extract_dates_from_text(text: str) -> List[str]:
    """
    Find potential dates with regex and parse them with dateutil for normalization.
    Returns list of ISO dates or human-readable dates.
    """
    # common patterns: YYYY-MM-DD, DD/MM/YYYY, Month Day, Year
    patterns = [
        r'\b(\d{4}-\d{2}-\d{2})\b',
        r'\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})\b',
        r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b',
        r'\b\d{1,2}\s+(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}\b'
    ]
    found = set()
    for pat in patterns:
        for m in re.findall(pat, text, flags=re.IGNORECASE):
            try:
                dt = dateparser.parse(m, fuzzy=True)
                if dt:
                    found.add(dt.date().isoformat())
                else:
                    found.add(m)
            except:
                found.add(m)
    # also try to parse free text like "on 12th October"
    extra = re.findall(r'\b(on|dated|dated:)\s+([A-Za-z0-9,\-\/\s]{3,30})', text, flags=re.IGNORECASE)
    for _, candidate in extra:
        try:
            dt = dateparser.parse(candidate, fuzzy=True)
            if dt:
                found.add(dt.date().isoformat())
        except:
            pass
    return sorted(list(found))

def extract_crime_types(text: str) -> List[str]:
    """Look for crime keywords (case-insensitive). Returns matched canonical keywords."""
    text_l = text.lower()
    matches = []
    for k in CRIME_KEYWORDS:
        if k in text_l:
            matches.append(k)
    # also check for specific section numbers like 'Section 302'
    sec_matches = re.findall(r'Section\s*[:\-]?\s*([0-9]{2,4})', text, flags=re.IGNORECASE)
    for s in sec_matches:
        matches.append(f"Section {s}")
    return list(dict.fromkeys(matches))  # unique, preserve order

def extract_ranked_names_with_regex(text: str) -> List[str]:
    """Capture patterns like 'SI Rajesh Kumar' or 'Inspector Amit Patel'"""
    found = []
    for m in re.finditer(TITLE_PATTERNS, text):
        title = m.group(1)
        name = m.group(2)
        entry = f"{title} {name}"
        found.append(entry)
    return found

def ner_extract_entities(text: str, ner_pipeline) -> Dict[str, List[str]]:
    """
    Use token-classification NER to extract PERSON/ORG/LOC.
    Returns dict: {"persons": [], "orgs": [], "locs": []}
    """
    try:
        ents = ner_pipeline(text[:2000])  # limit length; app can process long docs in chunks if needed
    except Exception:
        # fallback: empty
        ents = []
    persons, orgs, locs = [], [], []
    for e in ents:
        label = e.get("entity_group") or e.get("entity")
        word = e.get("word") or e.get("entity")
        if not word:
            continue
        if label and label.upper().startswith("PER"):
            persons.append(e["word"].strip())
        elif label and label.upper().startswith("ORG"):
            orgs.append(e["word"].strip())
        elif label and label.upper().startswith("LOC"):
            locs.append(e["word"].strip())
    # dedupe nicely
    return {
        "persons": list(dict.fromkeys([p for p in persons if p and len(p) > 1])),
        "orgs": list(dict.fromkeys([o for o in orgs if o and len(o) > 1])),
        "locs": list(dict.fromkeys([l for l in locs if l and len(l) > 1]))
    }

def fuzzy_lookup(candidate: str, choices: List[str], score_cutoff: int = 75) -> Optional[Tuple[str, int]]:
    """
    Fuzzy match candidate to choices using rapidfuzz.
    Returns best match and score if above cutoff.
    """
    if not choices:
        return None
    best = rf_process.extractOne(candidate, choices, scorer=fuzz.WRatio)
    if best and best[1] >= score_cutoff:
        # best = (match, score, index)
        return (best[0], int(best[1]))
    return None

def hybrid_extract_officer_info(text: str, ner_pipeline, datasets: dict, gazetteer: dict) -> Dict:
    """
    Hybrid approach:
      - NER (PERSON)
      - regex rank/title capture
      - gazetteer fuzzy matching (OFFICER_DATABASE + district reports)
      - departments via NER (ORG) + gazetteer fuzzy match
      - locations via NER (LOC) + gazetteer + direct matches
    Returns dict with extracted lists and match confidences.
    """
    # run NER
    ner_out = ner_extract_entities(text, ner_pipeline)
    persons = ner_out["persons"]
    orgs = ner_out["orgs"]
    locs = ner_out["locs"]

    # regex titles
    title_names = extract_ranked_names_with_regex(text)
    # gather candidates
    candidates = list(dict.fromkeys(title_names + persons))

    # build gazetteer pool
    officer_pool = list(gazetteer.get("names", []))
    dept_pool = list(gazetteer.get("departments", []))
    location_pool = list(gazetteer.get("locations", []))

    # also add any investigating officer ids/names from district reports if available
    dr = datasets.get("district_reports")
    if isinstance(dr, list):
        for rec in dr:
            # look for investigating_officer_id or officer name fields
            iid = rec.get("investigating_officer_id")
            if iid:
                officer_pool.append(iid)
            # if case report contains other structured names, add them
            nid = rec.get("police_station")
            if nid:
                dept_pool.append(nid)
    # dedupe pools
    officer_pool = list(dict.fromkeys([o for o in officer_pool if o]))
    dept_pool = list(dict.fromkeys([d for d in dept_pool if d]))
    location_pool = list(dict.fromkeys([l for l in location_pool if l]))

    found_officers = []
    found_officers_conf = []

    for cand in candidates:
        # try exact match first (case-insensitive)
        match = None
        for opt in officer_pool:
            if cand.lower() == opt.lower():
                match = (opt, 100)
                break
        if not match:
            # fuzzy
            f = fuzzy_lookup(cand, officer_pool)
            if f:
                match = f
        if match:
            found_officers.append(match[0])
            found_officers_conf.append({"name": match[0], "score": match[1], "source": "gazetteer/ner"})
        else:
            # keep original candidate if nothing matched, but mark as low confidence
            found_officers.append(cand)
            found_officers_conf.append({"name": cand, "score": 50, "source": "ner/raw"})

    # if no candidates from NER/reg, also scan text for last names in gazetteer with fuzzy small-window search
    if not found_officers:
        # consider matching whole gazetteer names within text
        for opt in officer_pool:
            if re.search(re.escape(opt.split()[-1]), text, flags=re.IGNORECASE):
                found_officers.append(opt)
                found_officers_conf.append({"name": opt, "score": 75, "source": "gazetteer-lastname"})

    # departments
    found_depts = []
    found_depts_conf = []
    # use ORG NER first
    for o in orgs:
        f = fuzzy_lookup(o, dept_pool, score_cutoff=60)
        if f:
            found_depts.append(f[0])
            found_depts_conf.append({"dept": f[0], "score": f[1], "source": "ner/org"})
        else:
            found_depts.append(o)
            found_depts_conf.append({"dept": o, "score": 50, "source": "ner/raw"})
    # fallback: look for dept keywords in text
    if not found_depts:
        for d in dept_pool:
            if d.lower() in text.lower():
                found_depts.append(d)
                found_depts_conf.append({"dept": d, "score": 90, "source": "gazetteer-exact"})

    # locations
    found_locs = []
    found_locs_conf = []
    for l in locs:
        f = fuzzy_lookup(l, location_pool, score_cutoff=60)
        if f:
            found_locs.append(f[0])
            found_locs_conf.append({"loc": f[0], "score": f[1], "source": "ner/loc"})
        else:
            found_locs.append(l)
            found_locs_conf.append({"loc": l, "score": 50, "source": "ner/raw"})
    if not found_locs:
        for lp in location_pool:
            if lp.lower() in text.lower():
                found_locs.append(lp)
                found_locs_conf.append({"loc": lp, "score": 85, "source": "gazetteer-exact"})

    # final de-dup
    found_officers = list(dict.fromkeys(found_officers)) or ["(Officer name not found - please specify)"]
    found_depts = list(dict.fromkeys(found_depts)) or ["(Department not specified)"]
    found_locs = list(dict.fromkeys(found_locs)) or []

    return {
        "officers": found_officers,
        "officers_conf": found_officers_conf,
        "departments": found_depts,
        "departments_conf": found_depts_conf,
        "locations": found_locs,
        "locations_conf": found_locs_conf
    }

# -----------------------
# Analysis pipeline
# -----------------------
def analyze_text(text: str, models: dict, datasets: dict, gazetteer: dict) -> Dict:
    """
    Main analysis: NER-based extraction, sentiment, summarization, crime/date detection.
    Returns a result dict that is JSON serializable.
    """
    ner = models["ner"]
    sentiment = models["sentiment"]
    summarizer = models["summarizer"]
    qa = models["qa"]

    # Basic clean
    text = text.strip()
    # Extract entities
    entities = hybrid_extract_officer_info(text, ner, datasets, gazetteer)
    # Sentiment on English text only
    try:
        sent_res = sentiment(text[:512])[0]
        normalized = sent_res["score"] if sent_res["label"] == "POSITIVE" else -sent_res["score"]
    except Exception:
        sent_res = {"label": "NEUTRAL", "score": 0.5}
        normalized = 0.0

    # Summary
    try:
        summary = text if len(text) < 120 else summarizer(text[:2000], max_length=150, min_length=40, do_sample=False)[0]["summary_text"]
    except Exception:
        # fallback heuristic: first 3 sentences
        summary = ". ".join(text.split(".")[:3]).strip() + "."

    # crime and dates
    crimes = extract_crime_types(text)
    dates = extract_dates_from_text(text)

    # recognition score (simple heuristic)
    base_score = (normalized + 1) / 2  # 0..1
    tag_boost = 0.0
    high_value = ["life_saving", "bravery", "de-escalation"]
    # compute tags heuristically from text keywords
    tags = []
    # quick tag heuristics
    if any(w in text.lower() for w in ["saved", "rescue", "life-saving", "revived"]):
        tags.append("life_saving")
    if any(w in text.lower() for w in ["brave", "courage", "heroic", "fearless"]):
        tags.append("bravery")
    if any(w in text.lower() for w in ["calm", "de-escalate", "mediation", "defused", "negotiation"]):
        tags.append("de-escalation")
    # boost
    for t in tags:
        if t in high_value:
            tag_boost += 0.12
    length_boost = min(0.1, len(text) / 1000 * 0.1)
    recognition_score = round(min(1.0, base_score + tag_boost + length_boost), 3)

    result = {
        "timestamp": datetime.now().isoformat(),
        "original_text": text,
        "summary": summary,
        "extracted_officers": entities["officers"],
        "extracted_departments": entities["departments"],
        "extracted_locations": entities["locations"],
        "officers_confidence": entities.get("officers_conf", []),
        "departments_confidence": entities.get("departments_conf", []),
        "locations_confidence": entities.get("locations_conf", []),
        "sentiment_label": sent_res.get("label", "NEUTRAL"),
        "sentiment_score": normalized,
        "suggested_tags": tags if tags else ["general_commendation"],
        "recognition_score": recognition_score,
        "text_length": len(text),
        "crime_tags": crimes,
        "dates_found": dates
    }
    return result

# -----------------------
# PDF summary creation
# -----------------------
def create_pdf_summary(result: Dict) -> BytesIO:
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle(
        'Title',
        parent=styles['Heading1'],
        fontSize=20,
        textColor=colors.HexColor('#1f4b99'),
        alignment=1,
        spaceAfter=12
    )
    heading_style = ParagraphStyle(
        'Heading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor('#2e6bb7'),
        spaceAfter=8
    )

    story.append(Paragraph("🚔 Police Recognition Report", title_style))
    story.append(Spacer(1, 0.2*inch))
    story.append(Paragraph(f"<b>Generated:</b> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}", styles['Normal']))
    story.append(Spacer(1, 0.15*inch))

    story.append(Paragraph("Summary", heading_style))
    story.append(Paragraph(result.get("summary", "(no summary)"), styles['Normal']))
    story.append(Spacer(1, 0.15*inch))

    metrics = [
        ["Metric", "Value"],
        ["Recognition Score", f"{result.get('recognition_score', 0)}/1.0"],
        ["Sentiment", result.get('sentiment_label', 'NEUTRAL')],
        ["Text Length", f"{result.get('text_length', 0)} characters"],
        ["Crime Tags", ", ".join(result.get('crime_tags', []) or ["None"])],
        ["Dates Found", ", ".join(result.get('dates_found', []) or ["None"])]
    ]
    table = Table(metrics, colWidths=[3*inch, 3*inch])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2e6bb7')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,1), (-1,-1), colors.beige)
    ]))
    story.append(table)
    story.append(Spacer(1, 0.2*inch))

    story.append(Paragraph("Identified Officers", heading_style))
    for o in result.get("extracted_officers", []):
        story.append(Paragraph(f"• {o}", styles['Normal']))
    story.append(Spacer(1, 0.12*inch))

    story.append(Paragraph("Departments", heading_style))
    for d in result.get("extracted_departments", []):
        story.append(Paragraph(f"• {d}", styles['Normal']))
    story.append(Spacer(1, 0.12*inch))

    story.append(Paragraph("Locations", heading_style))
    for l in result.get("extracted_locations", []):
        story.append(Paragraph(f"• {l}", styles['Normal']))
    story.append(Spacer(1, 0.12*inch))

    story.append(Paragraph("Suggested Tags", heading_style))
    for t in result.get("suggested_tags", []):
        story.append(Paragraph(f"• {t.replace('_', ' ').title()}", styles['Normal']))

    doc.build(story)
    buffer.seek(0)
    return buffer

# -----------------------
# Streamlit App UI
# -----------------------
def main():
    st.set_page_config(page_title="Police Recognition Analytics (EN-only)", layout="wide", initial_sidebar_state="expanded")
    st.title("🚔 Police Recognition Analytics")

    # Load datasets (optional)
    datasets = load_all_datasets()
    # Build gazetteer: try to read from OdishaIPCCrimedata.json if it has structure; else fallback to HARDCODED_OFFICERS
    gazetteer = {}
    # If OdishaIPCCrimedata.json has police strength / names, respect it
    odisha = datasets.get("odisha_ipc")
    if isinstance(odisha, dict) and "Angul" in odisha:
        # crude: try to pull officer names from keys if any
        gazetteer = HARDCODED_OFFICERS.copy()
    else:
        gazetteer = HARDCODED_OFFICERS.copy()

    # augment gazetteer with departments/stations from district reports
    dr = datasets.get("district_reports")
    if isinstance(dr, list):
        for rec in dr:
            ps = rec.get("police_station")
            if ps and ps not in gazetteer["departments"]:
                gazetteer["departments"].append(ps)
            io = rec.get("investigating_officer_id")
            if io and io not in gazetteer["names"]:
                gazetteer["names"].append(io)

    # Load Models
    with st.spinner("Loading models..."):
        models = load_models()

    if not models:
        st.error("Failed to load models.")
        return

    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        st.markdown("---")
        st.subheader("Data files found")
        for k, p in DATA_FILES.items():
            present = os.path.exists(p)
            st.markdown(f"- **{p}** — {'✅' if present else '❌ (missing)'}")
        st.markdown("---")
        st.subheader("Actions")
        if st.button("Clear Processed Data"):
            st.session_state.processed_data = []
            st.session_state.chat_history = []
            st.experimental_rerun()

    # Initialize session state
    if "processed_data" not in st.session_state:
        st.session_state.processed_data = []
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Process Feedback", "📊 Dashboard", "💬 Q&A Chat", "📥 Export Data"])

    # --- Tab1: Process Feedback ---
    with tab1:
        st.header("📝 Process New Feedback")
        col1, col2 = st.columns([2,1])
        with col1:
            input_method = st.radio("Input method", ["✍️ Text Input", "📄 File Upload"], horizontal=True)
            text_to_process = ""
            if input_method == "✍️ Text Input":
                text_to_process = st.text_area("Enter feedback, article, or document:", height=300, placeholder="Officer Smith from Central Police Station...")

            else:
                uploaded = st.file_uploader("Upload TXT or PDF", type=["txt","pdf"])
                if uploaded:
                    try:
                        if uploaded.type == "text/plain":
                            text_to_process = uploaded.getvalue().decode("utf-8", errors="ignore")
                            st.success(f"Loaded {len(text_to_process)} characters.")
                        elif uploaded.type == "application/pdf":
                            text_to_process = ""
                            with pdfplumber.open(uploaded) as pdf:
                                for page in pdf.pages:
                                    ptext = page.extract_text()
                                    if ptext:
                                        text_to_process += ptext + "\n"
                            st.success(f"Extracted {len(text_to_process)} characters from PDF.")
                    except Exception as e:
                        st.error(f"Error reading file: {e}")

            if text_to_process:
                with st.expander("Preview", expanded=False):
                    st.text_area("Preview", text_to_process[:1500], height=200)

        with col2:
            st.info("""
            - NER + gazetteer for officer extraction
            - Sentiment, summarization and export
            """)

            st.markdown("**Quick tips**")
            st.markdown("- For best extraction include rank (e.g., 'SI Rajesh Kumar') where possible.")
            st.markdown("- Upload PDFs with clear text (not image scans).")

        st.markdown("---")
        if st.button("🚀 Analyze Feedback"):
            if not text_to_process or not text_to_process.strip():
                st.warning("Please provide text or upload a TXT/PDF.")
            else:
                with st.spinner("Analyzing..."):
                    res = analyze_text(text_to_process, models, datasets, gazetteer)
                    st.session_state.processed_data.append(res)
                    st.success("Analysis complete.")
                    # Display summary cards
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Recognition Score", res["recognition_score"])
                    c2.metric("Sentiment", res["sentiment_label"])
                    c3.metric("Officers Found", len([o for o in res["extracted_officers"] if not o.startswith("(")]))
                    c4.metric("Text Length", res["text_length"])
                    st.markdown("---")
                    with st.expander("🔎 Details", expanded=True):
                        st.subheader("Summary")
                        st.write(res["summary"])
                        st.subheader("Officers")
                        for o in res["extracted_officers"]:
                            st.write(f"- {o}")
                        st.subheader("Departments")
                        for d in res["extracted_departments"]:
                            st.write(f"- {d}")
                        st.subheader("Locations")
                        for l in res["extracted_locations"]:
                            st.write(f"- {l}")
                        st.subheader("Crime Tags")
                        st.write(", ".join(res["crime_tags"]) if res["crime_tags"] else "None")
                        st.subheader("Dates Found")
                        st.write(", ".join(res["dates_found"]) if res["dates_found"] else "None")

                    # Export buttons
                    colp1, colp2 = st.columns(2)
                    with colp1:
                        pdf_buffer = create_pdf_summary(res)
                        st.download_button("📄 Download PDF Summary", data=pdf_buffer, file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf", mime="application/pdf")
                    with colp2:
                        st.download_button("📋 Download JSON", data=json.dumps(res, indent=2, ensure_ascii=False), file_name=f"data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", mime="application/json")

    # --- Tab2: Dashboard ---
    with tab2:
        st.header("📊 Dashboard")
        data = st.session_state.processed_data
        if not data:
            st.info("No processed items yet. Add some via 'Process Feedback' tab.")
        else:
            df = pd.DataFrame(data)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Items", len(df))
            col2.metric("Avg Recognition Score", f"{df['recognition_score'].mean():.2f}")
            pos_pct = (df['sentiment_label'] == 'POSITIVE').sum() / len(df) * 100
            col3.metric("Positive %", f"{pos_pct:.0f}%")
            total_officers = sum(len(x) for x in df['extracted_officers'])
            col4.metric("Total Officers Mentioned", total_officers)

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                st.subheader("Top Officers")
                all_offs = [o for officers in df['extracted_officers'] for o in officers if not o.startswith("(")]
                if all_offs:
                    series = pd.Series(all_offs).value_counts().head(10)
                    st.bar_chart(series)
                else:
                    st.write("No officers yet.")

            with c2:
                st.subheader("Tags")
                all_tags = [t for tags in df['suggested_tags'] for t in tags]
                if all_tags:
                    st.bar_chart(pd.Series(all_tags).value_counts())
                else:
                    st.write("No tags yet.")

            st.markdown("---")
            st.subheader("Recent Entries")
            for i in range(min(5, len(df))):
                row = df.iloc[-(i+1)]
                with st.expander(f"#{len(df)-i} | Score: {row['recognition_score']} | Sentiment: {row['sentiment_label']}"):
                    st.write("**Summary:**", row['summary'])
                    st.write("**Officers:**", ", ".join(row['extracted_officers']))
                    st.write("**Departments:**", ", ".join(row['extracted_departments']))
                    st.write("**Crime Tags:**", ", ".join(row['crime_tags']) if row['crime_tags'] else "None")
                    st.write("**Dates:**", ", ".join(row['dates_found']) if row['dates_found'] else "None")
                    if st.button(f"Download PDF for item #{len(df)-i}", key=f"dlpdf_{i}"):
                        pdfb = create_pdf_summary(row.to_dict())
                        st.download_button("Download", data=pdfb, file_name=f"report_item_{len(df)-i}.pdf", mime="application/pdf")

    # --- Tab3: Q&A Chat ---
    with tab3:
        st.header("💬 Q&A over Analyzed Data")
        if not st.session_state.processed_data:
            st.info("Process at least one feedback item first.")
        else:
            all_texts = " ".join([d['original_text'] for d in st.session_state.processed_data])
            question = st.text_input("Ask a question about the processed feedback:", key="qa_input")
            if st.button("Get Answer"):
                if question.strip():
                    with st.spinner("Searching..."):
                        try:
                            answer = models["qa"](question=question, context=all_texts[:3000])
                            ans_text = answer.get("answer", "No answer found.")
                        except Exception as e:
                            ans_text = f"Unable to answer: {e}"
                        st.session_state.chat_history.append({"q": question, "a": ans_text})
                        st.experimental_rerun()
            if st.session_state.chat_history:
                st.markdown("---")
                for ch in reversed(st.session_state.chat_history[-10:]):
                    st.markdown(f"**Q:** {ch['q']}")
                    st.markdown(f"**A:** {ch['a']}")
                    st.markdown("---")

    # --- Tab4: Export Data ---
    with tab4:
        st.header("📥 Export / Bulk Download")
        if not st.session_state.processed_data:
            st.info("No processed data.")
        else:
            df = pd.DataFrame(st.session_state.processed_data)
            st.subheader("Data Table")
            st.dataframe(df, height=400, use_container_width=True)
            st.markdown("---")
            csv = df.to_csv(index=False)
            json_data = df.to_json(orient="records", indent=2, force_ascii=False)
            st.download_button("Download CSV", data=csv, file_name=f"data_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
            st.download_button("Download JSON", data=json_data, file_name=f"data_{datetime.now().strftime('%Y%m%d')}.json", mime="application/json")

if __name__ == "__main__":
    main()
