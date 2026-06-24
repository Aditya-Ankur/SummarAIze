import streamlit as st
import pdfplumber
import docx
import io
import time
from openai import OpenAI, RateLimitError

background = """
<style>
[data-testid="stApp"]{{
    background-color: #000000;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
}}
</style>
"""

API_KEY = st.secrets["API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

st.markdown(background, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Model config
# ---------------------------------------------------------------------------
PRIMARY_MODEL  = "qwen/qwen3-next-80b-a3b-instruct:free"
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MAX_RETRIES    = 3   # attempts per model — total max API calls = 2 x MAX_RETRIES


# ---------------------------------------------------------------------------
# Shared API helper: flat loop, hard stop after 2 x MAX_RETRIES attempts
# ---------------------------------------------------------------------------
def call_api(messages):
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
                    temperature=1.0,
                    max_tokens=4096,
                )
                return response.choices[0].message.content   # success — exit immediately

            except RateLimitError as e:
                retry_after = 30
                try:
                    meta = e.body.get("error", {{}}).get("metadata", {{}})
                    retry_after = int(meta.get("retry_after_seconds", 30))
                except Exception:
                    pass

                is_last_attempt = (attempt == MAX_RETRIES - 1)
                is_last_model   = (model_index == len(models_to_try) - 1)

                if is_last_attempt:
                    break   # move to next model (or fall through to final error)

                # More retries left — show live countdown then retry same model
                label = st.empty()
                for remaining in range(retry_after, 0, -1):
                    label.warning(
                        f"⏳ Rate limited — retrying in **{{remaining}}s** "
                        f"(attempt {{attempt + 1}} of {{MAX_RETRIES}} on `{{model}}`)..."
                    )
                    time.sleep(1)
                label.empty()

            except Exception as e:
                st.error(f"❌ Unexpected API error: `{{e}}`")
                return None   # non-rate-limit error — stop immediately

    # Reached only when every attempt on every model was rate-limited
    total = MAX_RETRIES * len(models_to_try)
    msg = (
        f"❌ Both `{{PRIMARY_MODEL}}` and `{{FALLBACK_MODEL}}` are rate-limited "
        f"after {{MAX_RETRIES}} attempts each ({{total}} total tries). "
        "Please wait a minute and try again, or add a provider key at "
        "https://openrouter.ai/settings/integrations for higher limits."
    )
    st.error(msg)
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
# Prompts
# ---------------------------------------------------------------------------
def build_prompt(text, category):
    prompts = {{
        "general": (
            "Summarize and explain this document in simple words. "
            "Do not show any internal reasoning, chain-of-thought, or explanation "
            f"of your process:\n\n{{text}}"
        ),
        "legal": (
            "Summarize this legal document (this may be a legal notice too), "
            "highlighting key laws, sections involved, and judgements (and further "
            "actions to be taken by the receiver in case this is a legal notice; "
            "if this is not a legal notice, do not involve anything related to it). "
            "Do not show any internal reasoning, chain-of-thought, or explanation "
            f"of your process:\n\n{{text}}\n\n"
            "Keep it simple and easy to understand. Identify the laws and sections "
            "involved and what the receiver should do based on the judgements."
        ),
        "medical": (
            "Summarize this medical document/report by focusing on diagnosis, "
            "symptoms, test results, and further treatment plans (if mentioned). "
            "Do not show any internal reasoning, chain-of-thought, or explanation "
            f"of your process:\n\n{{text}}\n\n"
            "Keep it simple and easy to understand. Identify the disease and "
            "treatment plan. Mention medications, further tests, or surgeries if present."
        ),
    }}
    return prompts[category]


st.title("General PDF Summarization")

if "general_summary"   not in st.session_state: st.session_state.general_summary   = None
if "general_last_file" not in st.session_state: st.session_state.general_last_file = None


def summarize_text(text):
    messages = [
        {"role": "system", "content": "You are an AI model that summarizes text in simple words."},
        {"role": "user",   "content": build_prompt(text, "general")},
    ]
    return call_api(messages)

def ask_followup(summary, question):
    prompt = (
        f"Based on the following summary:\n\n{summary}\n\n"
        f"Please answer this question: {question}\n\n"
        "Do not show any internal reasoning, chain-of-thought, or explanation of your process. "
        "Answer in detailed, simple words without excessive jargons."
    )
    messages = [
        {"role": "system", "content": "You are an AI assistant that provides thoughtful answers based on the given summary."},
        {"role": "user",   "content": prompt},
    ]
    return call_api(messages)


st.markdown("Upload any PDF or DOCX to summarize it in simple words and explain it.")
uploaded_file = st.file_uploader("", type=["pdf", "docx"], key="general")

if uploaded_file:
    if st.session_state.general_last_file != uploaded_file.name:
        st.session_state.general_summary   = None
        st.session_state.general_last_file = uploaded_file.name

    text = extract_text(uploaded_file)
    if text:
        if st.session_state.general_summary is None:
            with st.spinner("Hol\'up! Let us Cook... \U0001f9d1\u200d\U0001f373"):
                st.session_state.general_summary = summarize_text(text)

        if st.session_state.general_summary:
            st.subheader("Summary/Explanation:")
            st.write(st.session_state.general_summary)
            st.subheader("Follow-up Questions")
            question = st.text_input("Ask a follow-up question based on the summary:")
            if question:
                with st.spinner("Thinking..."):
                    answer = ask_followup(st.session_state.general_summary, question)
                if answer:
                    st.subheader("Answer")
                    st.write(answer)
else:
    st.session_state.general_summary   = None
    st.session_state.general_last_file = None
