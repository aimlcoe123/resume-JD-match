import os
import re
import tempfile
import fitz
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from docx import Document
import spacy
from skillNer.general_params import SKILL_DB
from skillNer.skill_extractor_class import SkillExtractor
from spacy.matcher import PhraseMatcher

model = SentenceTransformer('BAAI/bge-base-en-v1.5')

nlp = spacy.load("en_core_web_lg")
skill_extractor = SkillExtractor(nlp, SKILL_DB, PhraseMatcher)


# === 1. File Extraction Functions ===
def extract_text_from_pdf(file):
    if hasattr(file, "read"):  # Detect if it's a BytesIO-like object
        file.seek(0)
        return "\n".join(page.get_text() for page in fitz.open(stream=file.read(), filetype="pdf"))
    else:  # If it's a file path
        return "\n".join(page.get_text() for page in fitz.open(file))

def extract_text(file):
    ext = os.path.splitext(file.name)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(file)
    else:
        raise ValueError(f"Unsupported file format: {ext}. Only PDF files are allowed.")

# === 2. Preprocessing Function ===
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# === 3. Chunking Function ===
def chunk_text(text, chunk_size=3):
    sentences = [s.strip() for s in re.split(r'(?<=[.!?]) +', text.strip()) if s.strip()]
    return [' '.join(sentences[i:i+chunk_size]) for i in range(0, len(sentences), chunk_size)]

# === 4. Semantic Similarity Score ===
def semantic_score(jd_text, resume_text):
    jd_chunks = chunk_text(jd_text)
    resume_chunks = chunk_text(resume_text)

    jd_embs = model.encode(jd_chunks, normalize_embeddings=True)
    resume_embs = model.encode(resume_chunks, normalize_embeddings=True)

    max_scores = []
    for jd_emb in jd_embs:
        cos_sim = util.cos_sim(jd_emb, resume_embs)[0]
        max_scores.append(cos_sim.max().item())

    return np.mean(max_scores)

# === 5. Keyword Matching Score (TF-IDF based) ===
def keyword_score(jd_text, resume_text):
    corpus = [jd_text, resume_text]
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    sim_matrix = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])
    return sim_matrix[0][0]

def extract_skills_from_jd(text):
    annotations = skill_extractor.annotate(text)
    return sorted(set(skill['doc_node_value'] for skill in annotations['results']['full_matches']))

def extract_skills_from_resume(text):
    annotations = skill_extractor.annotate(text)
    return sorted(set(skill['doc_node_value'] for skill in annotations['results']['full_matches']))

# === 6. Skill Overlap ===
def compute_skill_overlap(jd_text, resume_text, keyword_list):
    matches = [kw for kw in keyword_list if kw.lower() in resume_text.lower()]
    return len(matches) / len(keyword_list) if keyword_list else 0.0

# === 7. Hybrid Scoring Function ===
def hybrid_score(jd_text, resume_text, keyword_list):
    semantic_part = semantic_score(jd_text, resume_text)
    keyword_part = keyword_score(jd_text, resume_text)
    skill_overlap_part = compute_skill_overlap(jd_text, resume_text, keyword_list)

    if skill_overlap_part == 0:
        hybrid = 0
    else:
        hybrid = (semantic_part * semantic_part) + ((1 - semantic_part) * keyword_part) + (0.2 * skill_overlap_part)
    
    return hybrid, semantic_part, keyword_part, skill_overlap_part

def get_domain_keywords(domain):
    domain_keywords = {
        "data_engineer": ["data pipeline", "sql", "python", "big data", "etl", "spark", "hadoop", "data warehouse"],
        "hr": ["recruitment", "onboarding", "payroll", "employee benefits", "performance management", "hr software"],
        "software_engineer": ["coding", "python", "software design", "agile", "data structures", "algorithms", "frontend", "backend"],
        "marketing": ["digital marketing", "seo", "content strategy", "branding", "advertising", "social media", "market research"],
        "cloud_engineer": ["cloud computing", "aws", "azure", "gcp", "cloud architecture", "kubernetes", "docker", "infrastructure as code"]
    }
    return domain_keywords.get(domain.lower(), [])

def detect_domain(text):
    domains = ["data_engineer", "hr", "software_engineer", "marketing", "cloud_engineer"]
    domain_scores = {domain: 0 for domain in domains}
    
    # Assign a score based on keyword match for each domain
    for domain in domains:
        keywords = get_domain_keywords(domain)
        domain_scores[domain] = compute_skill_overlap(text, text, keywords)
    
    # Return the domain with the highest score
    best_domain = max(domain_scores, key=domain_scores.get)
    return best_domain

def highlight_missing_skills(jd_skills, resume_skills):
    # Normalize to lowercase for comparison
    jd_skills_lower = {skill.lower() for skill in jd_skills}
    resume_skills_lower = {skill.lower() for skill in resume_skills}

    print("\n=== JD Skills (Normalized) ===")
    print(jd_skills_lower)
    print("\n=== Resume Skills (Normalized) ===")
    print(resume_skills_lower)

    # Find skills present in JD but missing from Resume
    missing = jd_skills_lower - resume_skills_lower
    