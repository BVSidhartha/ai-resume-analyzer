import re
import numpy as np
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from PyPDF2 import PdfReader
from docx import Document

# -------------------------------
# Helper functions
# -------------------------------

STOPWORDS = set('''a an the and or but if while is are was were be been being
of in on at to from by for with about against between into through during before after
above below up down out over under again further then once here there when where why how all any both each few more most
other some such no nor not only own same so than too very can will just don should now
this that these those i me my we our you your he him his she her it its they them their as it\'s don\'t'''.split())

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9+/#.&-]+", " ", text)
    tokens = [t for t in text.split() if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)

def read_file(uploaded_file) -> str:
    name = uploaded_file.name.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(uploaded_file)
        return "\n".join([p.extract_text() or "" for p in reader.pages])
    elif name.endswith(".docx"):
        doc = Document(uploaded_file)
        return "\n".join([p.text for p in doc.paragraphs])
    else:
        return uploaded_file.read().decode("utf-8", errors="ignore")

def tfidf_cosine(resume_text: str, jd_text: str):
    docs = [clean_text(resume_text), clean_text(jd_text)]
    X = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95).fit_transform(docs)
    sim = float(cosine_similarity(X[0], X[1])[0][0])
    return sim

# -------------------------------
# Streamlit App
# -------------------------------

st.set_page_config(page_title="AI Resume Analyzer", page_icon="🧠", layout="wide")

st.title("🧠 AI Resume Analyzer")
st.write("Upload your resume and paste a job description to compare them.")

# Inputs
resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","txt"])
jd_text = st.text_area("Paste Job Description", height=250)

# Action
if st.button("Analyze", type="primary"):
    if not resume_file or not jd_text:
        st.warning("Please provide both a resume and a job description.")
        st.stop()

    resume_text = read_file(resume_file)
    sim = tfidf_cosine(resume_text, jd_text)

    st.metric("Similarity (TF-IDF cosine)", f"{sim:.3f}")
    st.success("Analysis complete ✅")
