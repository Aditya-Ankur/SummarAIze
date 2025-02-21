import streamlit as st
import pdfplumber
import docx
import io
from openai import OpenAI

if 'legal_summary' not in st.session_state:
    st.session_state.legal_summary = None

background = """
<style>
[data-testid="stApp"]{
    background-color: #000000;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
}
</style>
"""

st.title("Legal Document Summarization")

API_KEY = st.secrets["API_KEY"]

client = OpenAI(
    base_url = "https://openrouter.ai/api/v1",
    api_key = API_KEY
)

st.markdown(background, unsafe_allow_html=True)

def extract_text(uploaded_file):
    filetype=uploaded_file.name.split(".")[-1]
    if filetype=="pdf":
        return extract_text_from_pdf(uploaded_file)
    elif filetype=="docx":
        return extract_text_from_docx(uploaded_file)
    else:
        return print("!!Incompatible filetype!!")
    
def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(io.BytesIO(uploaded_file.read())) as pdf:
        return "\n" .join([page.extract_text() for page in pdf.pages if page.extract_text()]) 
    '''the above code line rnus a loop to go thorgh every pdf page and the n extracts text and joins them up '''

def extract_text_from_docx(uploaded_file):
    doc=docx.Document(uploaded_file)
    return "\n" .join([para.text for para in doc.paragraphs])



def summarize_text(text, category):
    prompts = {
        "general": f"Summarize and explain this document in simple words and do not show any internal reasoning, chain-of-thought, or explanation of your process:\n\n{text}",
        "legal": f"Summarize this legal document (this may be a legal notice too), highlighting key laws, sections involved, and judgements (and further actions to be taken by the reciever in case this is a legal notice.; if this is not a legal notice do not involve anything related to it) and do not show any internal reasoning, chain-of-thought, or explanation of your process:\n\n{text} and try to keep it simple and easy to understand. Try to identify the laws and sections involved and what the reciever should do based on the judgements and what the judgements mean.",
        "medical": f"Summarize this medical document/report by focusing on diagonsis, symptoms, tests results, and further treatment plas (if mentioned) and do not show any internal reasoning, chain-of-thought, or explanation of your process:\n\n{text} and try to keep it simple and easy to understand. Try to identify the disease and the treatment plan and what the patient should do based on the test results and what the test results mean. If there are any symptoms, mention them too. If there are any medications, mention them too. If there are any further tests, mention them too. If there are any surgeries, mention them too.",
    }
    
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "system", "content": "You are a AI model that summarizes the text in simple words."},
            {"role": "user", "content": prompts[category]}
        ],
        temperature=1.0,
        max_completion_tokens=128000,
    )

    output = response.choices[0].message.content
    return output

# ask follow up question
def ask_followup(summary, question):
    prompt = f"Based on the following summary:\n\n{summary}\n\n .Please answer the following Question: {question} and please do not show any internal reasoning, chain-of-thought, or explanation of your process. Answer it in detailed and simple words without too much legal jargons."
    response = client.chat.completions.create(
        model="deepseek/deepseek-r1:free",
        messages=[
            {"role": "system", "content": "You are an AI assistant that provides thoughtful answers based on the given summary."},
            {"role": "user", "content": prompt}
        ],
        temperature=1.0,
        max_completion_tokens=128000,
    )

    output = response.choices[0].message.content
    return output

st.markdown("Upload a legal document to summarize and ask questions about it.")
uploaded_file = st.file_uploader("Upload any PDF/DOCX", type=["pdf","docx"], key="legal")
if uploaded_file:
    text = extract_text(uploaded_file)
    if text:
        # Only generate summary if it hasn't been generated yet or if file changed
        if st.session_state.legal_summary is None:
            with st.spinner("Hol' up! Let us Cook... 🧑‍🍳"):
                st.session_state.legal_summary = summarize_text(text, "legal")
        
        st.subheader("Legal Summary/Explanation:")
        st.write(st.session_state.legal_summary)
        
        st.subheader("Follow-up Questions")
        question = st.text_input("Ask a follow-up question based on the summary:")
        if question:
            with st.spinner("Thinking..."):
                answer = ask_followup(st.session_state.legal_summary, question)
            st.subheader("Answer")
            st.write(answer)
else:
    # Reset the summary when no file is uploaded
    st.session_state.legal_summary = None