import streamlit as st


# custom background
background = """
<style>
[data-testid="stApp"]{
    background-color: #000000;
    opacity: 0.8;
    background-image:  repeating-radial-gradient( circle at 0 0, transparent 0, #000000 6px ), repeating-linear-gradient( #43434355, #434343 );
}
</style>
"""

st.markdown(background, unsafe_allow_html=True)

st.markdown("""
<style>
    .name-container {
        background: rgb(131,58,180);
        background: linear-gradient(90deg, rgba(131,58,180,1) 0%, rgba(253,29,29,1) 50%, rgba(252,176,69,1) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: inline;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .main-header {
        text-align: center;
        padding: 2rem 0;
        color: #ffffff;
        font-size: 2.5rem;
        font-weight: 600;
        margin-bottom: 1rem;
    }
    .description {
        text-align: center;
        color: #a0a0a0;
        font-size: 1.1rem;
        margin-bottom: 4rem;
        line-height: 1.6;
    }
    .custom-button-general, .custom-button-legal, .custom-button-medical {
        display: block;
        padding: 1.5rem;
        font-size: 1.5rem;
        font-weight: 500;
        text-align: center;
        text-decoration: none;
        border-radius: 8px;
        transition: all 0.3s ease;
        background: rgba(255, 255, 255, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #ffffff;
        backdrop-filter: blur(10px);
    }
    .custom-button-general:hover {
        transform: translateY(-2px);
        background: rgba(107, 235, 52, 0.69);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .custom-button-legal:hover {
        transform: translateY(-2px);
        background: rgba(3, 115, 252, 0.69);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .custom-button-medical:hover {
        transform: translateY(-2px);
        background: rgba(252, 3, 3, 0.69);
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    .feature-box {
        background: rgba(255, 255, 255, 0.05);
        padding: 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .feature-box h3 {
        color: #ffffff;
        font-size: 1.1rem;
        margin-bottom: 1rem;
        font-weight: 500;
    }
    .feature-box p {
        color: #a0a0a0;
        font-size: 0.9rem;
        margin: 0.5rem 0;
        line-height: 1.4;
    }
</style>
""", unsafe_allow_html=True)

# Main content
st.markdown('<h1 class="main-header">Summar<div class="name-container">AI</div>ze</h1>', unsafe_allow_html=True)

st.markdown('''
<p class="description">
    Transform your documents into clear, actionable insights with AI-powered summaries
</p>
''', unsafe_allow_html=True)

# Create three columns with equal spacing
col1, col2, col3 = st.columns([1, 1, 1])

with col1:
    st.markdown('''
        <a href="/General" class="custom-button-general" style="text-decoration: none; color:white;">
            General
        </a>
        <div class="feature-box">
            <h3>General Documents</h3>
            <p>• Quick PDF summaries</p>
            <p>• Key insights extraction</p>
            <p>• Comprehensive analysis</p>
        </div>
    ''', unsafe_allow_html=True)

with col2:
    st.markdown('''
        <a href="/Legal" class="custom-button-legal" style="text-decoration: none; color:white;">
            Legal
        </a>
        <div class="feature-box">
            <h3>Legal Documents</h3>
            <p>• Legal point extraction</p>
            <p>• Section highlights</p>
            <p>• Terms interpretation</p>
        </div>
    ''', unsafe_allow_html=True)

with col3:
    st.markdown('''
        <a href="/Medical" class="custom-button-medical" style="text-decoration: none; color:white;">
            Medical
        </a>
        <div class="feature-box">
            <h3>Medical Reports</h3>
            <p>• Report interpretation</p>
            <p>• Key findings summary</p>
            <p>• Treatment overview</p>
        </div>
    ''', unsafe_allow_html=True)