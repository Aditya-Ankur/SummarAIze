import streamlit as st
import pdfplumber
import docx
import io
import time
import json
from groq import Groq, RateLimitError

background = """
<style>
[data-testid="stApp"]{
    background-color: #000000;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
}
.compare-card {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.12);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 1rem;
}
.card-header {
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 0.6rem;
    letter-spacing: 0.02em;
}
.trajectory-badge {
    display: inline-block;
    padding: 0.4rem 1.2rem;
    border-radius: 999px;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.05em;
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
# Medical Report Comparison
# ---------------------------------------------------------------------------
def compare_reports(text_a, text_b):
    """
    Ask the LLM to perform a structured diff of two medical reports.
    Returns a JSON string that we parse into display cards.
    """
    prompt = f"""You are a medical analyst comparing two medical reports for the same patient.

REPORT A (Earlier / Baseline):
{text_a}

---

REPORT B (Later / Follow-up):
{text_b}

---

Carefully compare both reports and return your analysis as a JSON object with EXACTLY these keys:

{{
    "improved": ["list each finding or metric that improved from A to B — be specific"],
    "worsened": ["list each finding or metric that worsened from A to B — be specific"],
    "new_findings": ["new conditions, symptoms, or findings that appear in B but not in A"],
    "resolved": ["conditions, symptoms, or findings from A that are no longer present in B"],
    "medication_changes": ["any additions, removals, or dose changes in medications between A and B"],
    "overall_trajectory": "one of exactly: Improving / Worsening / Mixed / Stable / Insufficient Data",
    "trajectory_explanation": "2-3 sentence plain-language explanation of the overall health trajectory",
    "doctor_questions": ["3-5 specific questions the patient should ask their doctor based on these changes"]
}}

Rules:
- If a list has nothing to report, use an empty list [].
- Be specific — mention actual values or conditions where possible.
- Return ONLY the JSON object, no markdown, no explanation, no extra text.
"""
    messages = [
        {
            "role": "system",
            "content": "You are a medical document analyst. Return only a valid JSON object, no markdown."
        },
        {"role": "user", "content": prompt},
    ]
    return call_api(messages)


def parse_comparison(raw_response):
    """Parse the LLM JSON response, stripping markdown code fences if present."""
    if not raw_response:
        return None
    text = raw_response.strip()
    # Strip markdown code fences if the model added them
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first and last fence lines
        text = "\n".join(lines[1:-1]) if lines[-1].strip() == "```" else "\n".join(lines[1:])
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def render_list_card(title, items, color, icon):
    """Render a styled card for a list of findings."""
    if not items:
        return
    items_html = "".join(f"<li style='margin:0.3rem 0; color:#ddd;'>{item}</li>" for item in items)
    st.markdown(
        f"""
        <div class="compare-card" style="border-left: 4px solid {color};">
            <div class="card-header" style="color:{color};">{icon} {title}</div>
            <ul style="margin:0; padding-left:1.2rem;">{items_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_comparison(data):
    """Render the full comparison result as structured cards."""
    # Overall trajectory badge
    trajectory = data.get("overall_trajectory", "Insufficient Data")
    trajectory_colors = {
        "Improving":         ("#22c55e", "#052e16"),
        "Worsening":         ("#ef4444", "#2d0a0a"),
        "Mixed":             ("#f59e0b", "#2d1a00"),
        "Stable":            ("#3b82f6", "#0a1628"),
        "Insufficient Data": ("#6b7280", "#1a1a1a"),
    }
    fg, bg = trajectory_colors.get(trajectory, ("#6b7280", "#1a1a1a"))

    st.markdown(
        f"""
        <div style="text-align:center; margin: 1.5rem 0;">
            <div style="color:#aaa; font-size:0.9rem; margin-bottom:0.5rem;">OVERALL HEALTH TRAJECTORY</div>
            <span class="trajectory-badge" style="background:{bg}; color:{fg}; border: 2px solid {fg};">
                {trajectory}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Trajectory explanation
    explanation = data.get("trajectory_explanation", "")
    if explanation:
        st.info(f"📋 {explanation}")

    st.markdown("---")

    # Two-column layout for improved / worsened
    col1, col2 = st.columns(2)
    with col1:
        render_list_card("Improved", data.get("improved", []), "#22c55e", "✅")
    with col2:
        render_list_card("Worsened", data.get("worsened", []), "#ef4444", "❗")

    # Full-width cards for other categories
    render_list_card("New Findings", data.get("new_findings", []), "#f59e0b", "🆕")
    render_list_card("Resolved / No Longer Present", data.get("resolved", []), "#3b82f6", "✔️")
    render_list_card("Medication Changes", data.get("medication_changes", []), "#a78bfa", "💊")

    # Questions for doctor
    questions = data.get("doctor_questions", [])
    if questions:
        render_list_card("Questions to Ask Your Doctor", questions, "#38bdf8", "🩺")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
st.title("🔬 Medical Report Comparison")
st.markdown(
    "Upload two medical reports (e.g. an older and a newer one) to see what changed, "
    "what improved, what worsened, and what to ask your doctor."
)

st.markdown("---")

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("#### 📄 Report A — Earlier / Baseline")
    file_a = st.file_uploader("Upload Report A (PDF/DOCX)", type=["pdf", "docx"], key="compare_a")

with col_b:
    st.markdown("#### 📄 Report B — Later / Follow-up")
    file_b = st.file_uploader("Upload Report B (PDF/DOCX)", type=["pdf", "docx"], key="compare_b")

# Session state
if "compare_result" not in st.session_state:
    st.session_state.compare_result = None
if "compare_files" not in st.session_state:
    st.session_state.compare_files = (None, None)

if file_a and file_b:
    current_files = (file_a.name, file_b.name)

    # Reset if either file changes
    if st.session_state.compare_files != current_files:
        st.session_state.compare_result = None
        st.session_state.compare_files  = current_files

    st.markdown("---")
    compare_btn = st.button("🔍 Compare Reports", type="primary", use_container_width=True)

    if compare_btn or st.session_state.compare_result:
        if st.session_state.compare_result is None:
            text_a = extract_text(file_a)
            text_b = extract_text(file_b)

            if text_a and text_b:
                with st.spinner("Analysing and comparing reports... 🧑‍🔬"):
                    raw = compare_reports(text_a, text_b)
                    data = parse_comparison(raw)

                if data:
                    st.session_state.compare_result = data
                else:
                    # Fallback: show raw LLM response if JSON parsing fails
                    st.warning("⚠️ Could not parse structured comparison. Showing raw analysis:")
                    st.write(raw)
            else:
                st.error("Could not extract text from one or both files. Please check the uploads.")

        if st.session_state.compare_result:
            st.markdown("## 📊 Comparison Results")
            render_comparison(st.session_state.compare_result)

elif file_a and not file_b:
    st.info("📄 Report A uploaded. Now upload Report B to compare.")
elif file_b and not file_a:
    st.info("📄 Report B uploaded. Now upload Report A to compare.")
else:
    # Show instructional placeholder
    st.markdown(
        """
        <div style="
            border: 2px dashed rgba(255,255,255,0.15);
            border-radius: 12px;
            padding: 3rem;
            text-align: center;
            color: #666;
            margin-top: 2rem;
        ">
            <div style="font-size: 3rem;">🔬</div>
            <div style="font-size: 1.1rem; margin-top: 1rem;">Upload both reports above to begin comparison</div>
            <div style="font-size: 0.85rem; margin-top: 0.5rem;">
                Works best with blood reports, lab results, diagnostic reports, or any structured medical document
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
