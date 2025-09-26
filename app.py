import re
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="AI Resume Analyzer (text-only)", page_icon="🧠")

STOPWORDS = set('a an the and or but if while is are was were be been being of in on at to from by for with about against between into through during before after above below up down out over under again further then once here there when where why how all any both each few more most other some such no nor not only own same so than too very can will just don should now this that these those i me my we our you your he him his she her it its they them their as it\'s don\'t'.split())

def clean_text(t: str) -> str:
    t = t.lower()
    t = re.sub(r"[^a-z0-9+/#.&-]+", " ", t)
    return " ".join([w for w in t.split() if w not in STOPWORDS and len(w) > 1])

def tfidf_cosine(a: str, b: str) -> float:
    docs = [clean_text(a), clean_text(b)]
    X = TfidfVectorizer(ngram_range=(1,2), min_df=1, max_df=0.95).fit_transform(docs)
    return float(cosine_similarity(X[0], X[1])[0][0])

st.title("🧠 AI Resume Analyzer — text only (sanity)")
resume_text = st.text_area("Paste resume text", height=200)
jd_text = st.text_area("Paste job description", height=200)

if st.button("Analyze", type="primary"):
    if not resume_text or not jd_text:
        st.warning("Please paste both resume and JD text.")
        st.stop()
    sim = tfidf_cosine(resume_text, jd_text)
    st.metric("Similarity (TF-IDF cosine)", f"{sim:.3f}")
    st.success("Analysis complete ✅")

