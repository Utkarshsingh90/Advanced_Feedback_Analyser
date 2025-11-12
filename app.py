import streamlit as st
import pdfplumber
import docx
from textblob import TextBlob
import io
import re
import os
import requests
import textwrap
import fasttext
from transformers import pipeline, MarianMTModel, MarianTokenizer

# --- Page Configuration (MUST be the first st command) ---
st.set_page_config(
    page_title="Advanced Feedback Analysis",
    page_icon="🤖",
    layout="wide"
)

# --- MOCK DATABASE (Unchanged) ---
MOCK_OFFICERS_DB = {
    "smith": {"id": 123, "name": "Officer Smith", "unit_id": 14},
    "doe": {"id": 456, "name": "Detective Doe", "unit_id": 14},
    "davis": {"id": 789, "name": "Sgt. Davis", "unit_id": 15},
}
MOCK_UNITS_DB = {
    14: "14th Precinct",
    15: "Traffic Division (K-9)",
}
MOCK_TOPIC_KEYWORDS = {
    "compassion": "community_engagement", "kind": "community_engagement",
    "de-escalated": "de_escalation", "calmed the situation": "de_escalation",
    "fast response": "rapid_response", "arrived quickly": "rapid_response",
    "professional": "procedural_correctness",
}

# --- AI MODEL LOADING (NEW & IMPROVED) ---

# NEW: Download and cache the FastText model for language detection
@st.cache_resource
def load_fasttext_model():
    """Downloads and loads the FastText language ID model."""
    model_path = "lid.176.bin"
    if not os.path.exists(model_path):
        with st.spinner("Downloading language detection model (127MB)... This happens only once."):
            url = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
            try:
                r = requests.get(url, stream=True)
                r.raise_for_status()
                with open(model_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
            except Exception as e:
                st.error(f"Failed to download language model: {e}")
                return None
    try:
        return fasttext.load_model(model_path)
    except Exception as e:
        st.error(f"Failed to load language model: {e}")
        return None

@st.cache_resource
def load_translator(model_name):
    """Loads a specific translation model and tokenizer."""
    try:
        tokenizer = MarianTokenizer.from_pretrained(model_name)
        model = MarianMTModel.from_pretrained(model_name)
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading translation model {model_name}: {e}")
        return None, None

@st.cache_resource
def load_summarizer():
    """Loads the summarization pipeline."""
    try:
        return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")
    except Exception as e:
        st.error(f"Error loading summarization model: {e}")
        return None

@st.cache_resource
def load_qa_pipeline():
    """Loads the question-answering pipeline."""
    try:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except Exception as e:
        st.error(f"Error loading Q&A model: {e}")
        return None

# --- HELPER FUNCTIONS (NEW & IMPROVED) ---

def get_text_from_file(uploaded_file):
    """Extracts raw text from PDF, DOCX, or TXT files."""
    text = ""
    try:
        # Reset file pointer to the beginning
        uploaded_file.seek(0)
        if uploaded_file.type == "text/plain":
            text = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    text += page.extract_text(x_tolerance=1, y_tolerance=1) or ""
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            for para in doc.paragraphs:
                text += para.text + "\n"
    except Exception as e:
        st.error(f"Error reading file '{uploaded_file.name}': {e}")
        return None
    return text

# NEW: Stronger language detection
def detect_language(text, model):
    """Detects language using FastText and maps to 2-letter codes."""
    if not text or not model:
        return "en" # Default to English
    
    # FastText needs newline characters removed for best accuracy
    cleaned_text = text.replace("\n", " ").strip()
    if not cleaned_text:
        return "en"
        
    try:
        # k=1 means get the top 1 prediction
        predictions = model.predict(cleaned_text, k=1)
        if not predictions[0]:
            return "en"
            
        label = predictions[0][0] # e.g., '__label__ory_Orya'
        lang_code_3 = label.split('__')[-1].split('_')[0] # e.g., 'ory'
        
        # NEW: Map 3-letter (FastText) codes to 2-letter (Helsinki) codes
        lang_map = {
            "ory": "or",  # Odia
            "hin": "hi",  # Hindi
            "nep": "ne",  # Nepali
            "eng": "en",  # English
            # Add more mappings as needed
        }
        return lang_map.get(lang_code_3, "en") # Default to 'en' if not in map
    
    except Exception as e:
        st.warning(f"Language detection failed: {e}. Defaulting to English.")
        return "en"

# NEW: Robust chunking translator
def translate_to_english(text, lang):
    """Translates text to English if a model is available. Handles long text by chunking."""
    
    LANGUAGE_MODEL_MAP = {
        "hi": "Helsinki-NLP/opus-mt-hi-en",
        "or": "Helsinki-NLP/opus-mt-or-en",
        "ne": "Helsinki-NLP/opus-mt-ne-en",
    }
    
    if lang not in LANGUAGE_MODEL_MAP:
        st.warning(f"No translation model available for language '{lang}'. Processing in original language.")
        return text

    model_name = LANGUAGE_MODEL_MAP[lang]
    
    with st.spinner(f"Translating from '{lang}' to English..."):
        try:
            model, tokenizer = load_translator(model_name)
            if model is None:
                return text

            # --- NEW ROBUST CHUNKING LOGIC ---
            # Split text into chunks (e.g., by paragraphs or use textwrap)
            # We'll split by double newlines (paragraphs) as a start
            chunks = text.split('\n\n')
            chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
            
            if not chunks:
                return ""

            translated_chunks = []
            st.info(f"Translating long text in {len(chunks)} chunks...")
            
            for i, chunk in enumerate(chunks):
                # Update spinner
                st.spinner(f"Translating from '{lang}' to English... chunk {i+1}/{len(chunks)}")
                
                # Check if chunk is too long, and split it further if needed
                if len(chunk) > 1500: # 1500 chars is a safe proxy for ~512 tokens
                    sub_chunks = textwrap.wrap(chunk, 1500, break_long_words=True, replace_whitespace=False)
                else:
                    sub_chunks = [chunk]

                for sub_chunk in sub_chunks:
                    if not sub_chunk:
                        continue
                    
                    input_ids = tokenizer(sub_chunk, return_tensors="pt", padding=True, truncation=True, max_length=512).input_ids
                    translated_ids = model.generate(input_ids)
                    translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
                    translated_chunks.append(translated_text)
            
            return "\n\n".join(translated_chunks) # Re-join with paragraph breaks

        except Exception as e:
            st.error(f"Translation failed: {e}")
            return text

def process_text_pipeline(raw_text, lang_model):
    """
    The main AI pipeline that orchestrates all tasks.
    """
    processed_text = raw_text
    original_lang = "en"
    
    # --- 1. Language Detection & Translation ---
    try:
        original_lang = detect_language(raw_text, lang_model)
        if original_lang != "en":
            processed_text = translate_to_english(raw_text, original_lang)
    except Exception as e:
        st.warning(f"Language processing step failed: {e}")
        pass

    # --- 2. Dashboard Extraction ---
    # (Runs on the *translated* text)
    found_officer = None
    officer_name = None
    for keyword, officer_data in MOCK_OFFICERS_DB.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', processed_text, re.IGNORECASE):
            found_officer = officer_data
            officer_name = officer_data['name']; break
    
    suggested_officer_id = found_officer["id"] if found_officer else None
    suggested_unit_id = found_officer["unit_id"] if found_officer else None
    suggested_unit_name = MOCK_UNITS_DB.get(suggested_unit_id, "Unknown Unit") if suggested_unit_id else "N/A"
    
    suggested_tags = set()
    for keyword, tag in MOCK_TOPIC_KEYWORDS.items():
        if re.search(r'\b' + re.escape(keyword) + r'\b', processed_text, re.IGNORECASE):
            suggested_tags.add(tag)
    
    blob = TextBlob(processed_text)
    sentiment_score = blob.sentiment.polarity
    
    extracted_text = "No relevant snippet found."
    if officer_name:
        sentences = re.split(r'[.!?]', processed_text)
        for sentence in sentences:
            if re.search(r'\b' + re.escape(officer_name.split()[-1]) + r'\b', sentence, re.IGNORECASE):
                extracted_text = sentence.strip() + "..."; break
    elif len(processed_text) > 150:
         extracted_text = processed_text[:150].strip() + "..."
    elif processed_text:
        extracted_text = processed_text.strip()
    
    # --- 3. Summarization (NEW: with chunking) ---
    summary = "Summarization model failed or text too short."
    if len(processed_text.split()) > 40: # Only summarize if text is long enough
        try:
            summarizer = load_summarizer()
            if summarizer:
                # --- NEW CHUNKING LOGIC FOR SUMMARIZER ---
                # Chunk text to ~3000 chars (fits in 1024 tokens)
                chunks = textwrap.wrap(processed_text, 3000, break_long_words=True, replace_whitespace=False)
                st.info(f"Summarizing long text in {len(chunks)} chunks...")
                
                summaries = []
                for i, chunk in enumerate(chunks):
                    st.spinner(f"Summarizing chunk {i+1}/{len(chunks)}...")
                    summary_result = summarizer(chunk, max_length=150, min_length=25, do_sample=False)
                    summaries.append(summary_result[0]['summary_text'])
                
                # Join summaries with headings
                summary = f"--- SUMMARY (Part 1/{len(chunks)}) ---\n" + summaries[0]
                for i, s in enumerate(summaries[1:], start=2):
                    summary += f"\n\n--- SUMMARY (Part {i}/{len(chunks)}) ---\n" + s
            
        except Exception as e:
            st.error(f"Summarization failed: {e}")
            summary = "Summarization error."
    
    # --- Assemble Final Output ---
    output = {
        "original_text": raw_text,
        "processed_text": processed_text,
        "original_lang": original_lang,
        "summary": summary,
        "dashboard_details": {
            "extracted_text": extracted_text,
            "suggested_officer_id": suggested_officer_id,
            "suggested_unit_id": suggested_unit_id,
            "suggested_unit_name": suggested_unit_name,
            "suggested_sentiment": round(sentiment_score, 2),
            "suggested_tags": list(suggested_tags) or ["no_tags_found"]
        }
    }
    
    return output

# --- STREAMLIT APP UI (NEW: with on_change callbacks) ---

st.title("🤖 Advanced Feedback Analysis Pipeline")
st.markdown("This tool translates, analyzes, summarizes, and answers questions from community feedback.")

# Load the new language model
lang_model = load_fasttext_model()

# --- State Management ---
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None

# NEW: Callback to clear old results
def clear_old_results():
    """Wipes the session state to force a re-analysis."""
    st.session_state.analysis_result = None

# --- Sidebar (Unchanged) ---
with st.sidebar:
    st.title("ℹ️ App Guide")
    st.markdown("**1. Input:** Upload a file or paste text.")
    st.markdown("**2. Analyze:** Click the 'Analyze' button.")
    st.markdown("**3. Review:** Check the 'Dashboard', 'Summary', and 'Q&A' tabs.")
    st.markdown("---")
    st.subheader("Mock Database (For Demo)")
    with st.expander("Officers DB", expanded=False): st.json(MOCK_OFFICERS_DB)
    with st.expander("Units DB", expanded=False): st.json(MOCK_UNITS_DB)
    with st.expander("Topic Keywords", expanded=False): st.json(MOCK_TOPIC_KEYWORDS)

# --- Main App Body ---
input_tab1, input_tab2 = st.tabs(["📁 Upload a File", "📋 Paste Text"])
raw_text_input = None

with input_tab1:
    uploaded_file = st.file_uploader(
        "Upload a PDF, DOCX, or TXT file",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=False,
        on_change=clear_old_results # NEW: Clear results on new file
    )
    if uploaded_file:
        raw_text_input = get_text_from_file(uploaded_file)

with input_tab2:
    pasted_text = st.text_area(
        "Paste your text here (e.g., from a news article or email):", 
        height=250,
        on_change=clear_old_results # NEW: Clear results on new text
    )
    if pasted_text:
        raw_text_input = pasted_text

if st.button("Analyze Text", type="primary", use_container_width=True):
    if raw_text_input and lang_model:
        # Don't need to clear here, on_change already did
        with st.spinner("🧠 Starting full AI pipeline... This may take a moment."):
            st.session_state.analysis_result = process_text_pipeline(raw_text_input, lang_model)
    elif not raw_text_input:
        st.warning("Please upload a file or paste text first.")
    else:
        st.error("Language detection model failed to load. Cannot proceed.")

# --- Output Section (Unchanged, but Summarizer output will be different) ---
if st.session_state.analysis_result:
    st.markdown("---")
    st.success("Analysis Complete!")
    
    result = st.session_state.analysis_result
    
    out_tab1, out_tab2, out_tab3, out_tab4 = st.tabs([
        "📊 Dashboard Details", "📝 Summary", "❓ Ask a Question (Q&A)", "📜 Original vs. Translated"
    ])
    
    with out_tab1:
        st.subheader("Data for Recognition Dashboard")
        details = result['dashboard_details']
        sentiment = details['suggested_sentiment']
        if sentiment > 0.3: delta = f"{sentiment} (Good)"
        elif sentiment < -0.3: delta = f"{sentiment} (Bad)"
        else: delta = f"{sentiment}"
        st.metric(label="Sentiment", value="Positive" if sentiment > 0.3 else "Negative" if sentiment < -0.3 else "Neutral", delta=delta)
        
        col1, col2 = st.columns(2)
        with col1:
            st.text(f"Officer ID: {details['suggested_officer_id']}")
            st.text(f"Unit ID:    {details['suggested_unit_id']}")
            st.text(f"Unit Name:  {details['suggested_unit_name']}")
        with col2:
            st.text("Suggested Tags:")
            st.code(", ".join(details['suggested_tags']), language=None)
            
        st.subheader("Relevant Snippet")
        st.info(f"`{details['extracted_text']}`")
        if st.button("Approve & Send to Dashboard"):
            st.success("Approved! (This is a demo - no data was sent)")

    with out_tab2:
        st.subheader("AI-Generated Summary")
        # The summary will now be much longer, with headings
        st.info(result['summary'])

    with out_tab3:
        st.subheader("Ask a Question About the Text")
        context = result['processed_text']
        question = st.text_input("Ask something like 'Who was the officer?' or 'What was the outcome?'")
        if question:
            with st.spinner("Finding answer..."):
                try:
                    qa_pipeline = load_qa_pipeline()
                    if qa_pipeline:
                        qa_result = qa_pipeline(question=question, context=context)
                        st.success(f"**Answer:** {qa_result['answer']}")
                        st.caption(f"(Confidence: {qa_result['score']:.2f})")
                    else: st.error("Q&A model is not available.")
                except Exception as e: st.error(f"Q&A failed: {e}")

    with out_tab4:
        st.subheader("Text Processing")
        st.markdown(f"**Detected Language:** `{result['original_lang']}`")
        if result['original_lang'] != 'en':
            st.text_area("Original Text", result['original_text'], height=200)
            st.text_area("Translated Text (Used for Analysis)", result['processed_text'], height=200)
        else:
            st.info("Original text is in English. No translation needed.")
            st.text_area("Original Text", result['original_text'], height=200)