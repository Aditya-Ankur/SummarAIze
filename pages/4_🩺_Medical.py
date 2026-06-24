import streamlit as st
import pdfplumber
import docx
import io
import time
from openai import OpenAI, RateLimitError

background = """
<style>
[data-testid="stApp"]{
    background-color: #000000;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
}
</style>
"""

st.title("Medical Report Summarization")

API_KEY = st.secrets["API_KEY"]

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=API_KEY
)

st.markdown(background, unsafe_allow_html=True)

# --- Session state init ---
if "medical_summary" not in st.session_state:
    st.session_state.medical_summary = None
if "medical_last_file" not in st.session_state:
    st.session_state.medical_last_file = None


# --- Model config ---
PRIMARY_MODEL  = "qwen/qwen3-next-80b-a3b-instruct:free"
FALLBACK_MODEL = "meta-llama/llama-3.3-70b-instruct:free"
MAX_RETRIES    = 3   # number of retry attempts on 429 before switching to fallback


# --- Shared API call with retry + fallback ---
def call_api(messages, model=PRIMARY_MODEL, _attempt=0, _used_fallback=False):
    """
    Makes an OpenRouter chat completion call.
    - On 429: waits the suggested retry_after duration (live countdown), then retries.
    - After MAX_RETRIES failures on the primary model: switches to FALLBACK_MODEL.
    - Returns the response string, or None on total failure.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=1.0,
            max_tokens=4096,
        )
        return response.choices[0].message.content

    except RateLimitError as e:
        # --- Parse retry_after from OpenRouter metadata ---
        retry_after = 30  # safe default
        try:
            meta = e.body.get("error", {}).get("metadata", {})
            retry_after = int(meta.get("retry_after_seconds", 30))
        except Exception:
            pass

        if _attempt < MAX_RETRIES - 1:
            # --- Live countdown ---
            label = st.empty()
            for remaining in range(retry_after, 0, -1):
                label.warning(
                    f"⏳ Provider rate limit hit — retrying in **{remaining}s** "
                    f"(attempt {_attempt + 1} of {MAX_RETRIES})..."
                )
                time.sleep(1)
            label.empty()
            return call_api(messages, model=model, _attempt=_attempt + 1, _used_fallback=_used_fallback)

        elif not _used_fallback:
            # --- All retries on primary exhausted → try fallback ---
            st.warning(
                f"⚠️ `{model}` is still rate-limited after {MAX_RETRIES} retries. "
                f"Switching to fallback model `{FALLBACK_MODEL}`..."
            )
            return call_api(messages, model=FALLBACK_MODEL, _attempt=0, _used_fallback=True)

        else:
            st.error(
                f"❌ Both `{PRIMARY_MODEL}` and `{FALLBACK_MODEL}` are rate-limited right now.\n\n"
                f"Please wait a minute and try again, or add your own provider key at "
                f"https://openrouter.ai/settings/integrations to get higher limits."
            )
            return None

    except Exception as e:
        st.error(f"❌ Unexpected API error: `{e}`")
        return None


# --- Text extraction ---
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


# --- Summarization ---
def summarize_text(text, category):
    prompts = {
        "general": (
            f"Summarize and explain this document in simple words. "
            f"Do not show any internal reasoning, chain-of-thought, or explanation of your process:\n\n{text}"
        ),
        "legal": (
            f"Summarize this legal document (this may be a legal notice too), "
            f"highlighting key laws, sections involved, and judgements (and further actions to be taken "
            f"by the receiver in case this is a legal notice; if this is not a legal notice, do not involve "
            f"anything related to it). Do not show any internal reasoning, chain-of-thought, or explanation "
            f"of your process:\n\n{text}\n\nKeep it simple and easy to understand."
        ),
        "medical": (
            f"Summarize this medical document/report by focusing on diagnosis, symptoms, test results, "
            f"and further treatment plans (if mentioned). Do not show any internal reasoning, chain-of-thought, "
            f"or explanation of your process:\n\n{text}\n\n"
            f"Keep it simple and easy to understand. Identify the disease and treatment plan. "
            f"Mention what the patient should do based on test results, any medications, further tests, or surgeries."
        ),
    }

    messages = [
        {"role": "system", "content": "You are an AI model that summarizes text in simple words."},
        {"role": "user",   "content": prompts[category]},
    ]
    return call_api(messages)


# --- Follow-up Q&A ---
def ask_followup(summary, question):
    prompt = (
        f"Based on the following summary:\n\n{summary}\n\n"
        f"Please answer this question: {question}\n\n"
        f"Do not show any internal reasoning, chain-of-thought, or explanation of your process. "
        f"Answer in detailed, simple words without too many medical jargons."
    )
    messages = [
        {"role": "system", "content": "You are an AI assistant that provides thoughtful answers based on the given summary."},
        {"role": "user",   "content": prompt},
    ]
    return call_api(messages)


# --- UI ---
st.markdown("Upload a medical report to get a summary and ask questions about it.")

uploaded_file = st.file_uploader("Upload any PDF/DOCX", type=["pdf", "docx"], key="medical")

if uploaded_file:
    if st.session_state.medical_last_file != uploaded_file.name:
        st.session_state.medical_summary = None
        st.session_state.medical_last_file = uploaded_file.name

    text = extract_text(uploaded_file)

    if text:
        if st.session_state.medical_summary is None:
            with st.spinner("Hol' up! Let us Cook... 🧑‍🍳"):
                st.session_state.medical_summary = summarize_text(text, "medical")

        if st.session_state.medical_summary:
            st.subheader("Medical Summary/Explanation:")
            st.write(st.session_state.medical_summary)

            st.subheader("Follow-up Questions")
            question = st.text_input("Ask a follow-up question based on the summary:")
            if question:
                with st.spinner("Thinking..."):
                    answer = ask_followup(st.session_state.medical_summary, question)
                if answer:
                    st.subheader("Answer")
                    st.write(answer)
else:
    st.session_state.medical_summary = None
    st.session_state.medical_last_file = None
