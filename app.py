import streamlit as st
from summarizer_model import SummarizerModel
import fitz  # PyMuPDF for PDF reading

# ✅ Page config must come immediately after imports
st.set_page_config(page_title="AI Document Summarizer", page_icon="🤖", layout="centered")

# Load model (cached for performance)
@st.cache_resource
def load_model():
    return SummarizerModel()

summarizer = load_model()

# Function to extract text from PDF
def extract_text_from_pdf(uploaded_file):
    text = ""
    with fitz.open(stream=uploaded_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# 🎨 Animated title and custom CSS
st.markdown(
    """
    <style>
        .title {
            font-size: 40px;
            font-weight: 700;
            text-align: center;
            color: #4CAF50;
            animation: fadeIn 2s ease-in-out;
        }
        @keyframes fadeIn {
            from {opacity: 0;}
            to {opacity: 1;}
        }
        .footer {
            text-align: center;
            font-size: 14px;
            color: gray;
            margin-top: 20px;
        }
    </style>
    <div class="title">📄 AI Document Summarizer</div>
    """,
    unsafe_allow_html=True
)

st.write("Upload a PDF or paste your text below. Choose summary length and click **Summarize.**")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📂 Upload PDF", type=["pdf"])

with col2:
    summary_length = st.selectbox("🧩 Summary Length", ["short", "medium", "long"])

text_input = st.text_area("📝 Or paste text here", height=200)

if st.button("✨ Generate Summary"):
    with st.spinner("Generating AI Summary... ⏳"):
        text = ""
        if uploaded_file:
            text = extract_text_from_pdf(uploaded_file)
        elif text_input.strip():
            text = text_input.strip()
        else:
            st.error("Please upload a PDF or enter some text.")
            st.stop()

        summary = summarizer.summarize(text, summary_length)
        st.success("✅ Summary Generated Successfully!")
        st.markdown("### Summary Output:")
        st.write(summary)

        st.info(f"📄 Input length: {len(text)} characters | 🧠 Type: {summary_length.capitalize()}")

st.markdown('<div class="footer">Developed by Abhinandan Rajput 💚</div>', unsafe_allow_html=True)
