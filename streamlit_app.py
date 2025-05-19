import streamlit as st
from PIL import Image
import pandas as pd
from sentence_transformers import SentenceTransformer
from final_score import (
    extract_text, preprocess, extract_skills_from_jd,
    detect_domain, hybrid_score, extract_skills_from_resume
)
from st_aggrid import AgGrid, GridOptionsBuilder
import base64
from io import BytesIO

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Logo setup
logo = Image.open("image.png")
buffered = BytesIO()
logo.resize((50, 50)).save(buffered, format="PNG")
img_str = base64.b64encode(buffered.getvalue()).decode()
resized_icon = logo.resize((32, 64))

st.set_page_config(
    page_title="ATS CALCULATOR",
    page_icon=resized_icon
)

hide_streamlit_style = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

st.markdown(f"""
    <style>
        .block-container {{
            padding: 0 !important;
        }}

        .fixed-navbar {{
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 70px;
            background-color: #3b00dd;
            color: white;
            z-index: 9999;
            display: flex;
            align-items: center;
            padding: 0 1.5rem;
            box-shadow: 0px 2px 10px rgba(0, 0, 0, 0.2);
        }}

        .navbar-logo {{
            height: 40px;
            margin-left: 1rem;
        }}

        .navbar-title {{
            font-size: 24px;
            font-weight: bold;
            margin-left: 10px;
        }}

        .main-content {{
            padding-top: 90px;
            padding-left: 2rem;
            padding-right: 2rem;
        }}
        [data-testid="collapsedControl"] {{
            top: 85px !important;
        }}

        section[data-testid="stSidebar"] {{
            z-index: 1000;
            margin-top: 70px;
        }}
    </style>

    <div class="fixed-navbar">
        <img src="data:image/png;base64,{img_str}" class="navbar-logo" />
        <span class="navbar-title">ATS SCORE CALCULATOR</span>
    </div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Upload Files")
    jd_file = st.file_uploader("📑 Upload Job Description (PDF)", type="pdf")
    uploaded_resumes = st.file_uploader("📄 Upload Resumes (PDF)", type="pdf", accept_multiple_files=True)

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# Convert DOCX to PDF for processing
converted_resume_files = []
if uploaded_resumes:
    for file in uploaded_resumes:
        converted_resume_files.append(file)

if not converted_resume_files or not jd_file:
    st.info("Please upload resumes and job description using the sidebar.")
else:
    st.success("Files uploaded successfully. Calculating ATS scores...")

    jd_text_raw = extract_text(jd_file)
    jd_text_clean = preprocess(jd_text_raw)
    jd_skills = extract_skills_from_jd(jd_text_clean)

    results = []
    for resume_file in converted_resume_files:
        resume_name = resume_file.name.replace(".pdf", "").title()
        resume_text_raw = extract_text(resume_file)
        resume_text_clean = preprocess(resume_text_raw)

        resume_skills = extract_skills_from_resume(resume_text_clean)
        resume_domain = detect_domain(resume_text_clean)

        hybrid, sem, kw, skill_overlap = hybrid_score(jd_text_clean, resume_text_clean, model, jd_skills)

        def highlight_missing_skills(jd_skills, resume_skills):
            jd_skills_lower = {skill.lower() for skill in jd_skills}
            resume_skills_lower = {skill.lower() for skill in resume_skills}
            missing = jd_skills_lower - resume_skills_lower
            return sorted([skill.capitalize() for skill in missing])

        def get_matching_skills(jd_skills, resume_skills):
            jd_skills_lower = {skill.lower() for skill in jd_skills}
            resume_skills_lower = {skill.lower() for skill in resume_skills}
            match = jd_skills_lower & resume_skills_lower
            return sorted([skill.capitalize() for skill in match])

        missing_skills = highlight_missing_skills(jd_skills, resume_skills)
        matching_skills = get_matching_skills(jd_skills, resume_skills)

        results.append({
            "Candidate Name": resume_name,
            "Confidence Score (%)": f"{hybrid*100:.2f}",
            "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
            "Matching Skills": ", ".join(matching_skills) if matching_skills else "None"
        })

    df_main = pd.DataFrame(results)
    df_main['Confidence Score (%)'] = pd.to_numeric(df_main['Confidence Score (%)'], errors='coerce')
    df_main = df_main.sort_values(by='Confidence Score (%)', ascending=False)
    df_main['Rank'] = range(1, len(df_main) + 1)
    df_main = df_main[['Rank', 'Candidate Name', 'Confidence Score (%)', 'Missing Skills', 'Matching Skills']]

    # Dropdowns
    missing_skills_values = df_main['Missing Skills'].str.split(', ').explode().unique().tolist()
    matching_skills_values = df_main['Matching Skills'].str.split(', ').explode().unique().tolist()

    gb = GridOptionsBuilder.from_dataframe(df_main)
    gb.configure_default_column(wrapText=True, autoHeight=True)

    gb.configure_column('Missing Skills', 
                        cellEditor='agSelectCellEditor',
                        cellEditorParams={'values': missing_skills_values},
                        editable=True)

    gb.configure_column('Matching Skills', 
                        cellEditor='agSelectCellEditor',
                        cellEditorParams={'values': matching_skills_values},
                        editable=True)

    grid_options = gb.build()

    st.subheader("📊 ATS Score Summary")
    AgGrid(
        df_main,
        gridOptions=grid_options,
        enable_enterprise_modules=True,
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=False,
        height=500,
        theme="alpine"
    )

st.markdown('</div>', unsafe_allow_html=True)
