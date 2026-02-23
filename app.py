# =========================
# IMPORTS
# =========================
import streamlit as st
import matplotlib.pyplot as plt
import PyPDF2
import re
import nltk

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag


# =========================
# NLTK DOWNLOADS (run once)
# =========================
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("averaged_perceptron_tagger")


# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="📄",
    layout="wide"
)


# =========================
# SIDEBAR
# =========================
with st.sidebar:
    st.header("📌 About")
    st.info(
        """
        **AI Resume Analyzer & ATS Match Scorer**
        
        ✔ Resume vs Job Description Matching  
        ✔ ATS Compatibility Score  
        ✔ Skill Gap Detection  
        ✔ Resume Improvement Suggestions  
        """
    )

    st.header("⚙️ How It Works")
    st.write(
        """
        1. Upload your resume (PDF)  
        2. Paste job description  
        3. Click **Analyze Resume**  
        4. Get match score & insights  
        """
    )


# =========================
# HELPER FUNCTIONS
# =========================
def extract_text_from_pdf(uploaded_file):
    """Extract text from uploaded PDF"""
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
        return text
    except Exception as e:
        st.error(f"PDF Read Error: {e}")
        return ""


def clean_text(text):
    """Lowercase, remove special characters"""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def remove_stopwords(text):
    """Remove English stopwords"""
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    return " ".join([word for word in tokens if word not in stop_words])


def preprocess_text(text):
    """Full preprocessing pipeline"""
    return remove_stopwords(clean_text(text))


def calculate_similarity(resume_text, job_text):
    """TF-IDF + Cosine Similarity"""
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform([resume_text, job_text])
    score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
    return round(score * 100, 2)


def extract_keywords(text):
    """Extract important nouns & adjectives"""
    words = word_tokenize(text)
    tagged = pos_tag(words)
    keywords = [
        word for word, tag in tagged
        if tag.startswith(("NN", "JJ")) and len(word) > 2
    ]
    return set(keywords)


# =========================
# VISUALIZATION
# =========================
def display_gauge(score):
    fig, ax = plt.subplots(figsize=(6, 0.6))
    colors = ["#ff4b4b", "#ffa726", "#0f9d58"]
    color_index = min(int(score // 33), 2)

    ax.barh([0], [score], color=colors[color_index])
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("ATS Match Percentage")
    ax.set_title("Resume – Job Match Score")

    st.pyplot(fig)


# =========================
# MAIN APP
# =========================
def main():
    st.title("🚀 AI Resume Analyzer & ATS Match Scorer")
    st.markdown(
        """
        Upload your resume and job description to see  
        **how well your resume matches real-world ATS systems.**
        """
    )

    uploaded_file = st.file_uploader("📄 Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area("📝 Paste Job Description", height=200)

    if st.button("🔍 Analyze Resume"):
        if not uploaded_file or not job_description:
            st.warning("Please upload a resume and paste the job description.")
            return

        with st.spinner("Analyzing resume..."):
            resume_raw = extract_text_from_pdf(uploaded_file)
            if not resume_raw:
                return

            resume_processed = preprocess_text(resume_raw)
            job_processed = preprocess_text(job_description)

            # Similarity Score
            match_score = calculate_similarity(resume_processed, job_processed)

            # Keyword Analysis
            resume_keywords = extract_keywords(resume_processed)
            job_keywords = extract_keywords(job_processed)

            matched_skills = resume_keywords & job_keywords
            missing_skills = job_keywords - resume_keywords

        # =========================
        # RESULTS
        # =========================
        st.subheader("📊 Analysis Results")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("ATS Match Score", f"{match_score}%")
        with col2:
            st.progress(match_score / 100)

        display_gauge(match_score)

        # Feedback Message
        if match_score < 40:
            st.error("❌ Low Match — Resume needs significant improvement.")
        elif match_score < 70:
            st.warning("⚠️ Moderate Match — Resume can be optimized.")
        else:
            st.success("✅ Excellent Match — Resume is highly ATS compatible!")

        # =========================
        # KEYWORD INSIGHTS
        # =========================
        st.subheader("🧠 Skill Gap Analysis")

        col3, col4 = st.columns(2)
        with col3:
            st.markdown("**✅ Matched Skills**")
            st.write(", ".join(list(matched_skills)[:15]) or "None")

        with col4:
            st.markdown("**❌ Missing Skills**")
            st.write(", ".join(list(missing_skills)[:15]) or "None")

        # Resume Health Score
        health_score = max(0, 100 - len(missing_skills))
        st.metric("🩺 Resume Health Score", f"{health_score}/100")

        # Suggestions
        st.subheader("💡 Improvement Suggestions")
        if missing_skills:
            for skill in list(missing_skills)[:5]:
                st.write(f"➕ Consider adding experience or projects related to **{skill}**")
        else:
            st.success("Your resume already covers key job requirements!")


# =========================
# RUN APP
# =========================
if __name__ == "__main__":
    main()