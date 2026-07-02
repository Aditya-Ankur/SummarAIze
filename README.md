# 🩺⚖️ SummarAIze

**Your AI-Powered Assistant for Medical and Legal Document Intelligence**

SummarAIze is an advanced document intelligence application designed to transform complex, jargon-heavy medical and legal documents into clear, actionable, and easy-to-understand insights. Built with Python, Streamlit, and modern Generative AI capabilities.

## ✨ Features

* **🩺 Medical Report Summarization & Q&A:** Upload medical reports (PDF/DOCX) to get a clear summary of diagnoses, symptoms, test results, and treatment plans. Ask follow-up questions to understand your health better, completely free of medical jargon.
* **⚖️ Legal Document Analysis & Q&A:** Upload legal documents, notices, or contracts to receive summaries highlighting key laws, sections, and necessary actions. Ask specific questions about your rights and obligations.
* **🔬 Medical Report Comparison:** Upload two medical reports (e.g., past and current) to automatically generate a detailed comparative analysis. Track changes in health markers, spot improvements or risks, and log medication updates.
* **🧠 Retrieval-Augmented Generation (RAG):** Employs `sentence-transformers` and cosine similarity-based retrieval to ensure the AI's answers are strictly grounded in your uploaded documents, reducing hallucinations and context limits.
* **🛡️ Hallucination Guard:** A strict verification layer that double-checks if the AI's answers are genuinely derivable from the source text, ensuring maximum reliability for sensitive domains.

## 👥 Team
* **Vansh Shrivas**
* **Abhinandan Roshan**
* **Adarsh Pandey**
* **Aditya Samant**

## 🎥 Demo
Check out a demonstration of the app in action:
> **Note:** This video demo, recorded by teammate Abhinandan, showcases an older version of the application. The current version has been significantly upgraded with a RAG pipeline, Hallucination Guard, and the Report Comparison tool.
> 
> **[Watch the Demo on YouTube](https://youtu.be/OFXJNnFRNB0)**

## 🚀 Live App
The app is deployed on Streamlit Community Cloud. 
> *Note: The live link may occasionally be unavailable if API usage limits have been reached.*
> 
> **[Try SummarAIze Live](https://summaraize-ex06bc.streamlit.app/)**

## 🛠️ Local Installation

To run the app locally, you will need a Groq API key.

1. **Clone the repository:**
   ```shell
   git clone https://github.com/VanshShrivas/SummarAIze-Your-AI-assistant-for-medical-and-legal-purposes.git
   cd SummarAIze-Your-AI-assistant-for-medical-and-legal-purposes
   ```

2. **Install dependencies:**
   ```shell
   pip install -r requirements.txt
   ```

3. **Set up Secrets:**
   Create a `.streamlit/secrets.toml` file in the root directory and add your API key:
   ```toml
   GROQ_API_KEY = "your_api_key_here"
   ```

4. **Run the application:**
   ```shell
   streamlit run SummarAIze.py --server.fileWatcherType none
   ```

## 🏗️ Tech Stack
* **Frontend/Framework:** Streamlit
* **LLM Engine:** Groq API (Llama 3.3 70B / Mixtral 8x7B)
* **Embeddings & RAG:** `sentence-transformers` (all-MiniLM-L6-v2), `scikit-learn` (Cosine Similarity), `numpy`
* **Document Parsing:** `pdfplumber`, `python-docx`
