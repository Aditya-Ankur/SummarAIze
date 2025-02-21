import streamlit as st

# API_KEY = st.secrets["API_KEY"]

st.set_page_config(page_title="SummarAIze", layout="wide")

# Redirect to Home page
st.markdown("""
    <meta http-equiv="refresh" content="0; url=/Home">
    <script>window.location.href = "/Home"</script>
""", unsafe_allow_html=True)