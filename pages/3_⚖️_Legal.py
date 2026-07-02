import streamlit as st
import pdfplumber
import docx
import io
import time
import numpy as np
from groq import Groq, RateLimitError
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

background = """
<style>
[data-testid="stApp"]{
    background-color: #000000;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
}
</style>
"""

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

st.markdown(background, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
PRIMARY_MODEL  = "llama-3.3-70b-versatile"
FALLBACK_MODEL = "mixtral-8x7b-32768"
MAX_RETRIES    = 3


# ---------------------------------------------------------------------------
# Embedding model — loaded once per app session, cached globally
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading embedding model (first time only)...")
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


# ---------------------------------------------------------------------------
# API helper
# ---------------------------------------------------------------------------
def call_api(messages, temperature=0.2, max_tokens=4096):
    models_to_try = [PRIMARY_MODEL, FALLBACK_MODEL]

    for model_index, model in enumerate(models_to_try):
        if model_index > 0:
            st.warning(
                f"⚠️ `{PRIMARY_MODEL}` exhausted after {MAX_RETRIES} retries. "
                f"Switching to fallback model `{FALLBACK_MODEL}`..."
            )

        for attempt in range(MAX_RETRIES):
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content

            except RateLimitError as e:
                retry_after = 30
                try:
                    retry_after = int(e.response.headers.get("retry-after", 30))
                except Exception:
                    pass

                is_last_attempt = (attempt == MAX_RETRIES - 1)
                if is_last_attempt:
                    break

                label = st.empty()
                for remaining in range(retry_after, 0, -1):
                    label.warning(
                        f"⏳ Rate limited — retrying in **{remaining}s** "
                        f"(attempt {attempt + 1} of {MAX_RETRIES} on `{model}`)..."
                    )
                    time.sleep(1)
                label.empty()

            except Exception as e:
                st.error(f"❌ Unexpected API error: `{e}`")
                return None

    total = MAX_RETRIES * len(models_to_try)
    st.error(
        f"❌ Both `{PRIMARY_MODEL}` and `{FALLBACK_MODEL}` are rate-limited "
        f"after {MAX_RETRIES} attempts each ({total} total tries). "
        "Please wait a minute and try again."
    )
    return None


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------
def extract_text(uploaded_file):
    filetype = uploaded_file.name.split(".")[-1].lower()
    if filetype == "pdf":
        return extract_text_from_pdf(uploaded_file)
    elif filetype == "docx":
        return extract_text_from_docx(uploaded_file)
    else:
        st.error("Incompatible file type. Please upload a PDF or DOCX file.")
        return ""

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n".join(
            [page.extract_text() for page in pdf.pages if page.extract_text()]
        )

def extract_text_from_docx(uploaded_file):
    doc = docx.Document(uploaded_file)
    return "\n".join([para.text for para in doc.paragraphs])


# ---------------------------------------------------------------------------
# RAG: Chunking → Embedding → Retrieval
# ---------------------------------------------------------------------------
def chunk_text(text, chunk_size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

def build_rag_index(text):
    model = load_embedding_model()
    chunks = chunk_text(text)
    with st.spinner("🔍 Indexing document for smart retrieval..."):
        embeddings = model.encode(chunks, show_progress_bar=False)
    return chunks, embeddings

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=5):
    model = load_embedding_model()
    q_embedding = model.encode([question])
    scores = cosine_similarity(q_embedding, embeddings)[0]
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [chunks[i] for i in top_indices]


# ---------------------------------------------------------------------------
# Hallucination Guard — verify answer is grounded in retrieved context
# ---------------------------------------------------------------------------
def check_grounding(answer, context_chunks):
    context = "\n---\n".join(context_chunks)
    messages = [
        {
            "role": "system",
            "content": "You are a fact-checker. Reply ONLY with GROUNDED or NOT_GROUNDED."
        },
        {
            "role": "user",
            "content": (
                f"Document excerpts:\n{context}\n\n"
                f"Answer to verify:\n{answer}\n\n"
                "Is this answer derivable solely from the document excerpts above? "
                "Reply only: GROUNDED or NOT_GROUNDED"
            ),
        },
    ]
    try:
        response = client.chat.completions.create(
            model=PRIMARY_MODEL,
            messages=messages,
            temperature=0,
            max_tokens=10,
        )
        result = response.choices[0].message.content.strip().upper()
        return "NOT_GROUNDED" not in result
    except Exception:
        return True  # fail open


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------
def summarize_text(text):
    prompt = (
        "Summarize this legal document (this may be a legal notice too), "
        "highlighting key laws, sections involved, and judgements (and further "
        "actions to be taken by the receiver in case this is a legal notice; "
        "if this is not a legal notice, do not involve anything related to it). "
        "Do not show any internal reasoning, chain-of-thought, or explanation "
        f"of your process:\n\n{text}\n\n"
        "Keep it simple and easy to understand. Identify the laws and sections "
        "involved and what the receiver should do based on the judgements."
    )
    messages = [
        {"role": "system", "content": "You are an AI model that summarizes legal documents in simple, clear language."},
        {"role": "user",   "content": prompt},
    ]
    return call_api(messages)


# ---------------------------------------------------------------------------
# RAG-powered Follow-up Q&A
# ---------------------------------------------------------------------------
def ask_followup(question, chunks, embeddings):
    relevant_chunks = retrieve_relevant_chunks(question, chunks, embeddings)
    context = "\n\n---\n\n".join(relevant_chunks)

    prompt = (
        "Using ONLY the following excerpts from the legal document:\n\n"
        f"{context}\n\n"
        f"Answer this question: {question}\n\n"
        "If the answer cannot be found in the excerpts, respond with exactly: "
        "'This information is not available in the uploaded document.'\n"
        "Answer in simple, clear language without excessive legal jargon."
    )
    messages = [
        {
            "role": "system",
            "content": "You are a legal document assistant. Answer questions strictly based on the provided document excerpts only."
        },
        {"role": "user", "content": prompt},
    ]
    answer = call_api(messages)
    is_grounded = check_grounding(answer, relevant_chunks) if answer else True
    return answer, is_grounded


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("⚖️ Legal Document Summarization")

# Session state initialisation
for key in ["legal_summary", "legal_last_file", "legal_chunks", "legal_embeddings"]:
    if key not in st.session_state:
        st.session_state[key] = None

st.markdown("Upload a legal document to summarize and ask questions about it.")
uploaded_file = st.file_uploader("Upload any PDF/DOCX", type=["pdf", "docx"], key="legal")

if uploaded_file:
    # Reset state when a new file is uploaded
    if st.session_state.legal_last_file != uploaded_file.name:
        st.session_state.legal_summary    = None
        st.session_state.legal_last_file  = uploaded_file.name
        st.session_state.legal_chunks     = None
        st.session_state.legal_embeddings = None

    text = extract_text(uploaded_file)
    if text:
        # Generate summary once
        if st.session_state.legal_summary is None:
            with st.spinner("Hol'up! Let us Cook... 🧑‍🍳"):
                st.session_state.legal_summary = summarize_text(text)

        # Build RAG index once
        if st.session_state.legal_chunks is None:
            chunks, embeddings = build_rag_index(text)
            st.session_state.legal_chunks     = chunks
            st.session_state.legal_embeddings = embeddings

        if st.session_state.legal_summary:
            st.subheader("Legal Summary/Explanation:")
            st.write(st.session_state.legal_summary)

            st.subheader("Follow-up Questions")
            question = st.text_input("Ask a question about your legal document:")

            if question:
                    with st.spinner("Searching document and generating answer..."):
                        answer, is_grounded = ask_followup(
                            question,
                            st.session_state.legal_chunks,
                            st.session_state.legal_embeddings,
                        )

                    if answer:
                        st.subheader("Answer")
                        st.write(answer)

                        # Hallucination badge
                        if is_grounded:
                            st.success("✅ Answer grounded in your document")
                        else:
                            st.warning(
                                "⚠️ This answer may go beyond what is in your document — "
                                "please verify with a qualified lawyer."
                            )
else:
    st.session_state.legal_summary    = None
    st.session_state.legal_last_file  = None
    st.session_state.legal_chunks     = None
    st.session_state.legal_embeddings = None
