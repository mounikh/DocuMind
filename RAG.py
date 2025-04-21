import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from sentence_transformers import SentenceTransformer
import docx
import fitz  # PyMuPDF

# --- PAGE CONFIG ---
st.set_page_config(page_title="DocuMind - Ask Your File", page_icon="📄", layout="wide")

# --- 🌙 DARK MODE ---
def apply_custom_theme():
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    st.sidebar.markdown("## ⚙️ Settings")
    toggle = st.sidebar.checkbox("🌙 Enable Dark Mode", value=st.session_state.dark_mode)

    if toggle:
        st.session_state.dark_mode = True
        st.markdown("""
            <style>
                html, body, [class*="css"] {
                    background-color: #1e1e1e !important;
                    color: white !important;
                }
                .stTextInput input, .stFileUploader label, .stButton button {
                    background-color: #333 !important;
                    color: white !important;
                }
            </style>
        """, unsafe_allow_html=True)
    else:
        st.session_state.dark_mode = False

apply_custom_theme()

# --- LOAD MODELS ---
@st.cache_resource
def load_models():
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    tokenizer = AutoTokenizer.from_pretrained("deepset/roberta-base-squad2")
    model = AutoModelForQuestionAnswering.from_pretrained("deepset/roberta-base-squad2")
    return embedder, tokenizer, model

embedder, tokenizer, qa_model = load_models()
qa_pipeline = pipeline('question-answering', model=qa_model, tokenizer=tokenizer)

# --- SESSION STATE INIT ---
if "qa_history" not in st.session_state:
    st.session_state.qa_history = []

if "full_text" not in st.session_state:
    st.session_state.full_text = ""

# --- HEADER ---
st.markdown("<h1 style='text-align: center;'>📄 DocuMind</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Ask questions from your uploaded document 🔍</h4>", unsafe_allow_html=True)
st.markdown("---")

# --- SIDEBAR ACTIONS ---
if st.sidebar.button("🆕 New Chat"):
    st.session_state.qa_history = []
    st.session_state.full_text = ""
    st.experimental_rerun()

uploaded_file = st.file_uploader("📁 Upload a document (.txt / .docx / .pdf)", type=['txt', 'docx', 'pdf'])

# --- LOAD TEXT ---
def load_text(file):
    try:
        if file.name.endswith(".txt"):
            return file.read().decode("utf-8")
        elif file.name.endswith(".docx"):
            doc = docx.Document(file)
            return "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
        elif file.name.endswith(".pdf"):
            with fitz.open(stream=file.read(), filetype="pdf") as doc:
                return "".join(page.get_text() for page in doc)
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
        return None

# --- CONTEXT CHUNKING ---
def get_context(question, full_text, top_n=5, chunk_size=600, overlap=100):
    words = full_text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk) > 50:
            chunks.append(chunk)
    if not chunks:
        return []
    
    top_n = min(top_n, len(chunks))
    question_embedding = embedder.encode([question], convert_to_tensor=True)
    chunk_embeddings = embedder.encode(chunks, convert_to_tensor=True)
    similarities = torch.nn.functional.cosine_similarity(question_embedding, chunk_embeddings)
    top_idx = torch.topk(similarities, top_n).indices.tolist()
    return [chunks[i] for i in top_idx]

# --- MAIN LOGIC ---
if uploaded_file:
    st.session_state.full_text = load_text(uploaded_file)

if st.session_state.full_text:
    st.success("✅ Document uploaded successfully!")
    question = st.text_input("💬 Ask a question from the document:")

    if question:
        with st.spinner("🔎 Searching for the best answer..."):
            top_contexts = get_context(question, st.session_state.full_text, top_n=5)
            if not top_contexts:
                st.warning("⚠️ Not enough content to answer. Try a longer document.")
            else:
                answers, scores = [], []
                for context in top_contexts:
                    result = qa_pipeline(question=question, context=context)
                    answers.append(result['answer'])
                    scores.append(result['score'])

                best_idx = scores.index(max(scores))
                st.markdown("### 📌 Top Matching Contexts")
                for i, ctx in enumerate(top_contexts, 1):
                    st.markdown(f"**Context {i}:**")
                    st.info(ctx)

                st.markdown("### ✅ Answers")
                for i, (ans, score) in enumerate(zip(answers, scores), 1):
                    confidence = f"{score:.4f}"
                    if i - 1 == best_idx:
                        st.markdown(f"🌟 **Best Answer {i}:** `{ans}`")
                    else:
                        st.markdown(f"**Answer {i}:** `{ans}`")
                    st.markdown(f"🔍 **Confidence:** `{confidence}`")

                # Add to history
                st.session_state.qa_history.append({
                    "question": question,
                    "answer": answers[best_idx],
                    "score": scores[best_idx]
                })

# --- SIDEBAR HISTORY ---
if st.session_state.qa_history:
    st.sidebar.markdown("## 🕑 Question History")
    for item in reversed(st.session_state.qa_history):
        st.sidebar.markdown(f"**Q:** {item['question']}")
        st.sidebar.markdown(f"A: `{item['answer']}`")
        st.sidebar.markdown(f"🔍 _Confidence:_ `{item['score']:.4f}`")
