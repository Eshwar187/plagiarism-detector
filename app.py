import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pdfminer.high_level
import io

# ------------------ Load Model ------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")

model = load_model()

# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="Plagiarism Checker", layout="wide")
st.title("📄 Plagiarism Detection System")
st.write("Upload a document to check for plagiarism (semantic + multilingual).")

# Threshold slider
threshold = st.slider("Set Similarity Threshold", 0.5, 0.95, 0.85, 0.01)

uploaded_file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])

if uploaded_file:
    # ------------------ Extract Text ------------------
    if uploaded_file.name.endswith(".pdf"):
        text = pdfminer.high_level.extract_text(uploaded_file)
    else:
        text = uploaded_file.read().decode("utf-8")

    # Split into sentences
    sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 5]

    if len(sentences) < 2:
        st.warning("Not enough text to analyze plagiarism.")
    else:
        embeddings = model.encode(sentences)
        sim_matrix = cosine_similarity(embeddings)

        # ------------------ Plagiarism Detection ------------------
        plagiarized = []
        for i in range(len(sentences)):
            for j in range(i+1, len(sentences)):
                if sim_matrix[i][j] > threshold:
                    plagiarized.append((sentences[i], sentences[j], sim_matrix[i][j]))

        plag_percent = (len(plagiarized) / len(sentences)) * 100

        # ------------------ Show Results ------------------
        st.subheader("📊 Plagiarism Report")
        st.write(f"**Total Sentences:** {len(sentences)}")
        st.write(f"**Plagiarized Pairs Found:** {len(plagiarized)}")
        st.write(f"**Plagiarism Percentage:** {plag_percent:.2f}%")

        st.markdown("---")

        # Highlight sentences
        for s1, s2, score in plagiarized:
            st.markdown(f"<p style='color:red;'>⚠️ Sentence 1: {s1}</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:orange;'>➡️ Sentence 2: {s2}</p>", unsafe_allow_html=True)
            st.markdown(f"🔗 Similarity Score: **{score:.2f}**")
            st.write("---")

        # ------------------ Downloadable Report ------------------
        if plagiarized:
            report = io.StringIO()
            report.write("Plagiarism Report\n\n")
            report.write(f"Total Sentences: {len(sentences)}\n")
            report.write(f"Plagiarized Pairs Found: {len(plagiarized)}\n")
            report.write(f"Plagiarism Percentage: {plag_percent:.2f}%\n\n")

            for s1, s2, score in plagiarized:
                report.write(f"Sentence 1: {s1}\n")
                report.write(f"Sentence 2: {s2}\n")
                report.write(f"Similarity Score: {score:.2f}\n")
                report.write("-" * 50 + "\n")

            st.download_button(
                label="📥 Download Report",
                data=report.getvalue(),
                file_name="plagiarism_report.txt",
                mime="text/plain"
            )
