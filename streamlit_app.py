import plotly.express as px

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

from reportlab.lib.styles import getSampleStyleSheet
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st
import pdfplumber
import tempfile
import joblib
import re
import pandas as pd
import os

# =========================
# ENV FIX
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="REAL INDUSTRY ATS SYSTEM", layout="wide")

# =========================
# MODEL LOADING (SAFE)
# =========================
@st.cache_resource(show_spinner=True)
def load_models():
    clf = joblib.load("resume_classifier.pkl")
    tfidf = joblib.load("tfidf_vectorizer.pkl")

    semantic_model = SentenceTransformer(
        "all-MiniLM-L6-v2",
        cache_folder="./hf_cache"
    )

    return clf, tfidf, semantic_model

clf, tfidf, semantic_model = load_models()

# =========================
# SKILLS DB
# =========================
skills_db = pd.read_csv("skills.csv")["skill"].dropna().tolist()

# =========================
# CLEAN SKILL VALIDATION (FIXED FAKE SKILLS BUG)
# =========================
def is_valid_skill(skill):
    if not skill:
        return False

    skill = skill.strip().lower()

    # REMOVE noise like AR, R, Go, C
    if len(skill) <= 2:
        return False

    return True

# =========================
# SKILL FAMILY (INDUSTRY LOGIC BOOST)
# =========================
SKILL_FAMILY = {
    "machine learning": ["ml", "ai", "deep learning"],
    "nlp": ["text mining", "language processing"],
    "cloud": ["aws", "azure", "gcp"],
    "speech ai": ["whisper", "wav2vec2", "speech recognition"],
    "llm": ["transformer", "hugging face", "generative ai"]
}

def soft_match(skill, jd_skills):
    skill_l = skill.lower()

    for key, values in SKILL_FAMILY.items():
        if key in skill_l or skill_l in values:
            for jd in jd_skills:
                jd_l = jd.lower()
                if jd_l == key or jd_l in values:
                    return True
    return False

# =========================
# SKILL EXTRACTION (CLEAN + STRICT)
# =========================
def extract_skills(text, skills_db):
    text = text.lower()
    found = set()

    for skill in skills_db:
        skill_clean = skill.lower().strip()

        # STRICT MATCH ONLY (no hallucination)
        pattern = r'\b' + re.escape(skill_clean) + r'\b'

        if re.search(pattern, text):
            found.add(skill)

    return sorted(found)

# =========================
# SEMANTIC MATCHING (FILTERED REALISTIC ATS)
# =========================
def semantic_match(resume_skills, jd_skills):

    if not resume_skills or not jd_skills:
        return [], set()

    matches = []
    matched_jd = set()

    resume_emb = semantic_model.encode(resume_skills)
    jd_emb = semantic_model.encode(jd_skills)

    for i, r in enumerate(resume_skills):

        if not is_valid_skill(r):
            continue

        for j, jd in enumerate(jd_skills):

            if not is_valid_skill(jd):
                continue

            # REMOVE noisy tiny matches
            if len(r) <= 2 or len(jd) <= 2:
                continue

            sim = cosine_similarity(
                [resume_emb[i]],
                [jd_emb[j]]
            )[0][0]

            # STRICT THRESHOLD (REAL ATS STYLE)
            if sim >= 0.78:
                matches.append((r, jd, round(sim * 100, 2)))
                matched_jd.add(jd)

    return matches, matched_jd

# =========================
# ATS SCORING ENGINE (INDUSTRY LEVEL)
# =========================
def ats_score(resume_skills, jd_skills, semantic_matches):

    if not jd_skills:
        return 0

    exact = set(resume_skills) & set(jd_skills)
    semantic = set([j for _, j, _ in semantic_matches])

    score = 0
    max_score = len(jd_skills) * 2

    soft_bonus = 0

    for r in resume_skills:
        for jd in jd_skills:
            if soft_match(r, jd):
                soft_bonus += 0.3

    for skill in jd_skills:

        weight = 2 if skill.lower() in [
            "python", "machine learning", "nlp"
        ] else 1

        if skill in exact:
            score += weight

        elif skill in semantic:
            score += 0.8

        else:
            score += min(soft_bonus, 0.2)

    final = (score / max_score) * 100
    return round(min(final, 100), 2)

# =========================
# PDF TEXT EXTRACTION
# =========================
def extract_text(pdf):
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(pdf.read())
        path = tmp.name

    text = ""

    with pdfplumber.open(path) as p:
        for page in p.pages:
            text += (page.extract_text() or "") + " "

    return text
# =========================
# PDF REPORT GENERATOR
# =========================
def generate_report(score, matched, missing, role):

    pdf_file = "ATS_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "AI Resume Parser ATS Report",
            styles["Title"]
        )
    )

    content.append(
        Paragraph(
            f"Predicted Role: {role}",
            styles["BodyText"]
        )
    )

    content.append(
        Paragraph(
            f"ATS Score: {score}%",
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            "Matched Skills:",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            ", ".join(matched),
            styles["BodyText"]
        )
    )

    content.append(Spacer(1, 12))

    content.append(
        Paragraph(
            "Missing Skills:",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            ", ".join(missing),
            styles["BodyText"]
        )
    )

    doc.build(content)

    return pdf_file
# =========================
# STREAMLIT UI
# =========================
st.markdown(
    """
    <h1 style='text-align:center;
               color:#FF69B4;'>
    🚀 AI Resume Parser & ATS Analyzer
    </h1>
    """,
    unsafe_allow_html=True
)

file = st.file_uploader("Upload Resume PDF", type=["pdf"])

if file:

    text = extract_text(file)

    st.subheader("📄 Resume Text")
    st.text_area("Preview", text[:2500], height=300)

    # =========================
    # ROLE PREDICTION
    # =========================
    vec = tfidf.transform([text])
    category = clf.predict(vec)[0]

    st.subheader("🎯 Predicted Role")
    st.success(category)
    emails = re.findall(
        r'[\w\.-]+@[\w\.-]+',
        text
    )  

    if emails:
        st.subheader("📧 Email")
        st.write(emails[0])

    phone_match = re.search(        
        r'(\+91[\s-]?)?[6-9]\d{9}',
        text
    )

    if phone_match:
        st.subheader("📱 Phone")
        st.write(phone_match.group())

    linkedin = re.findall(
        r'linkedin\.com\/in\/[A-Za-z0-9\-]+',
        text,
        re.IGNORECASE
    )

    if linkedin:
        st.subheader("🔗 LinkedIn")
        st.write(linkedin[0])
    # =========================
    # SKILLS
    # =========================
    resume_skills = extract_skills(text, skills_db)

    st.subheader("🛠 Skills Found")
    st.write(", ".join(resume_skills))
    strength = 0

    strength += min(
        len(resume_skills) * 3,
        40
    )

    if "Machine Learning" in resume_skills:
        strength += 10

    if "Python" in resume_skills:
        strength += 10

    if "Data Analytics" in resume_skills:
        strength += 10

    if "NLP" in resume_skills:
        strength += 10

    strength = min(strength, 100)

    st.subheader("💪 Resume Strength")
    st.progress(strength / 100)
    st.write(f"{strength}/100")
    summary = f"""
    Candidate suitable for {category} roles.
    Strong skills in {', '.join(resume_skills[:5])}.
    Resume Strength Score: {strength}/100.
    """

    st.info(summary)
    # =========================
    # JD INPUT
    # =========================
    jd_text = st.text_area("Paste Job Description")

    if jd_text:

        jd_skills = extract_skills(jd_text, skills_db)

        semantic_matches, _ = semantic_match(
            resume_skills,
            jd_skills
        )

        final_score = ats_score(
            resume_skills,
            jd_skills,
            semantic_matches
        )

        matched = sorted(
           list(
                set(resume_skills) &
                set(jd_skills)
           )
        )

        missing = sorted(
           list(
                set(jd_skills) -
            	 set(resume_skills)
           )
        )
    # =========================
    # OUTPUT
    # =========================
        st.subheader("📊 REAL ATS SCORE")
        st.metric("Match Score", f"{final_score}%")
        st.progress(final_score / 100)
        if final_score >= 80:
            rank = "A"
        elif final_score >= 60:
            rank = "B"
        elif final_score >= 40:
            rank = "C"
        else:
            rank = "D"

        st.subheader("🏆 Resume Rank")
        st.success(rank)

        st.subheader("📋 JD Skills")
        st.write(", ".join(jd_skills) if jd_skills else "None")

        st.subheader("✅ Matched Skills")
        st.write(", ".join(matched) if matched else "None")

        st.subheader("❌ Missing Skills")
        st.write(", ".join(missing))
        skill_gap = (
            len(missing) /
            len(jd_skills)
        ) * 100

        st.subheader("📉 Skill Gap")
        st.metric(
            "Gap %",
            f"{skill_gap:.1f}%"
        )
        st.subheader("📚 Recommended Skills To Learn")

        for skill in missing[:5]:
            st.write(f"• {skill}")
        st.subheader("📈 Dashboard Analytics")

        chart_data = pd.DataFrame({
            "Category": ["Matched", "Missing"],
            "Count": [len(matched), len(missing)]
        })

        fig = px.pie(
             chart_data,
             values="Count",
             names="Category",
             title="Skill Match Analysis",
             color="Category",
             color_discrete_map={
                 "Matched": "#87CEEB",   # Sky Blue
                 "Missing": "#FF69B4"    # Pink
             }
         )

        st.plotly_chart(
             fig,
             use_container_width=True
         )
        st.subheader("🧠 Semantic Matches (FILTERED)")
        if semantic_matches:
            for r, j, s in semantic_matches:
                st.write(f"{r} → {j} ({s}%)")
        else:
            st.write("No semantic matches")

        # =========================
        # DECISION ENGINE
        # =========================
        st.subheader("🎯 Hiring Decision")

        if final_score >= 85:
            st.success("🔥 Strong Hire")
        elif final_score >= 70:
            st.info("👍 Good Candidate")
        elif final_score >= 50:
            st.warning("⚠️ Trainable Candidate")
        else:
            st.error("❌ Not Suitable")
        pdf_file = generate_report(
            final_score,
            list(matched),
            list(missing),
            category
        )

        with open(pdf_file, "rb") as f:

            st.download_button(
                  "📄 Download ATS Report",
                  f,
                  file_name="ATS_Report.pdf"
            )