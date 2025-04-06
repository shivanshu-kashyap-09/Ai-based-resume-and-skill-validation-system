import pyttsx3
import speech_recognition as sr
from sentence_transformers import SentenceTransformer, util
import fitz  # PyMuPDF for PDF processing
import docx
import requests
import re
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Initialize models
similarity_model = SentenceTransformer('all-MiniLM-L6-v2')

model_name = r"E:\models\flan-t5-base"
 
llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
llama_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


# Text-to-speech
engine = pyttsx3.init()
def speak(text):
    engine.say(text)
    engine.runAndWait()

# Speech-to-text
def listen():
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        recognizer.adjust_for_ambient_noise(source, duration=1)

        try:
            audio = recognizer.listen(source, timeout=60, phrase_time_limit=30000)
            print("Capturing your answer...")
            response = recognizer.recognize_google(audio)
            print("Answer:", response)
            return response
        except sr.WaitTimeoutError:
            print("You didn’t say anything. Moving to next question.")
            return "No answer provided."
        except sr.UnknownValueError:
            print("Couldn't understand the audio.")
            return "No answer provided."
        except sr.RequestError:
            print("STT API error.")
            return "No answer provided."



# GitHub Access
GITHUB_API_TOKEN = "GITHUB_ACCESS_TOKEN"

GITHUB_PATTERN = r"https://github.com/([a-zA-Z0-9-_]+)"
GITHUB_REPO_PATTERN = r"https://github.com/([a-zA-Z0-9-]+)/([a-zA-Z0-9-]+)"
LINKEDIN_PATTERN = r"https://www.linkedin.com/in/[a-zA-Z0-9_-]+"

# ---------------------- UTIL FUNCTIONS ----------------------
def extract_clickable_links(file_path):
    links = {"github_profile": None, "github_repo": None, "linkedin": None}
    ext = os.path.splitext(file_path)[-1].lower()

    if ext == ".pdf":
        doc = fitz.open(file_path)
        for page in doc:
            for link in page.get_links():
                uri = link.get("uri", "")
                if re.match(GITHUB_REPO_PATTERN, uri):
                    links["github_repo"] = uri
                elif re.match(GITHUB_PATTERN, uri):
                    links["github_profile"] = uri
                elif re.match(LINKEDIN_PATTERN, uri):
                    links["linkedin"] = uri

    elif ext == ".docx":
        doc = docx.Document(file_path)
        for rel in doc.part.rels.values():
            if "hyperlink" in rel.reltype:
                url = rel.target_ref
                if re.match(GITHUB_REPO_PATTERN, url):
                    links["github_repo"] = url
                elif re.match(GITHUB_PATTERN, url):
                    links["github_profile"] = url
                elif re.match(LINKEDIN_PATTERN, url):
                    links["linkedin"] = url

        for para in doc.paragraphs:
            if not links["github_profile"] and re.search(GITHUB_PATTERN, para.text):
                links["github_profile"] = re.search(GITHUB_PATTERN, para.text).group(0)
            if not links["github_repo"] and re.search(GITHUB_REPO_PATTERN, para.text):
                links["github_repo"] = re.search(GITHUB_REPO_PATTERN, para.text).group(0)
            if not links["linkedin"] and re.search(LINKEDIN_PATTERN, para.text):
                links["linkedin"] = re.search(LINKEDIN_PATTERN, para.text).group(0)

    return links

def fetch_github_repos(github_profile_url):
    username = github_profile_url.rstrip('/').split('/')[-1]
    url = f"https://api.github.com/users/{username}/repos"
    headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    return [{"name": repo["name"], "url": repo["html_url"]} for repo in response.json()] if response.status_code == 200 else []

def fetch_repo_languages(github_repo_url):
    owner, repo = github_repo_url.rstrip('/').split('/')[-2:]
    url = f"https://api.github.com/repos/{owner}/{repo}/languages"
    headers = {"Authorization": f"token {GITHUB_API_TOKEN}"}
    response = requests.get(url, headers=headers)
    return {lang.lower(): percent for lang, percent in response.json().items()} if response.status_code == 200 else {}

def extract_text_from_pdf(resume_path):
    doc = fitz.open(resume_path)
    return "\n".join([page.get_text("text") for page in doc]).strip()

def extract_potential_languages(text):
    return set(re.findall(r'\b[a-zA-Z#++]+\b', text.lower()))

# ------------------ SCORING & MATCH ------------------
def compute_similarity(resume_text, job_description):
    resume_embedding = similarity_model.encode(resume_text, convert_to_tensor=True)
    job_embedding  = similarity_model.encode(job_description, convert_to_tensor=True)
    similarity_score = util.pytorch_cos_sim(resume_embedding, job_embedding).item()
    return round(similarity_score * 100, 2)

def compute_project_name_match_score(resume_text, github_repos):
    match_count = sum(1 for repo in github_repos if repo['name'].lower() in resume_text.lower())
    return round((match_count / len(github_repos)) * 100 if github_repos else 0, 2)

# Extract programming languages dynamically
def extract_potential_languages(text):
    """Extract potential programming languages dynamically from text."""
    words = set(re.findall(r'\b[a-zA-Z#++]+\b', text.lower()))  # Extract words with letters, '#', or '++'
    return words

def compute_language_match_score(resume_text, job_description, github_repos):
    resume_langs = extract_potential_languages(resume_text)
    job_langs = extract_potential_languages(job_description)
    repo_languages_set = set()

    for repo in github_repos:
        repo_langs = fetch_repo_languages(repo['url'])  # returns a dict
        repo_languages_set.update(lang.lower() for lang in repo_langs.keys())

    # Step 1: Match Repo Languages with Resume Skills
    repo_resume_match = repo_languages_set & resume_langs

    # Step 2: Match Repo Languages with Job Description
    repo_job_match = repo_languages_set & job_langs

    # Scoring
    repo_resume_score = round((len(repo_resume_match) / len(repo_languages_set)) * 100, 2) if repo_languages_set else 0
    repo_job_score = round((len(repo_job_match) / len(job_langs)) * 100, 2) if job_langs else 0

    final_score = round((repo_resume_score + repo_job_score) / 2, 2)

    return final_score

# ------------------ INTERVIEW PART ------------------
def generate_questions_with_llama(resume_text, job_description, github_repos):
    repo_text = "\n".join([f"{repo['name']} - {repo['url']}" for repo in github_repos])

    prompt = f"""
You are an AI technical interviewer.

Generate exactly 5 unique technical interview questions:
- The first 2 should be based ONLY on the candidate's *resume*.
- The last 3 should be based on the *GitHub repositories* and *job description*.

Resume:
{resume_text}

Job Description:
{job_description}

GitHub Repositories:
{repo_text}

Only return the questions numbered from 1 to 5.
"""

    try:
        input_ids = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
        output = llama_model.generate(**input_ids, max_new_tokens=300, do_sample=True, temperature=0.7)
        result = llama_tokenizer.decode(output[0], skip_special_tokens=True)

        print("🧠 Raw model output:\n", result)

        lines = result.strip().split('\n')
        questions = []
        for line in lines:
            match = re.match(r"^\s*\d+[).:-]\s*(.*)", line)
            if match:
                questions.append(match.group(1).strip())

        questions = questions[:5]

        if len(questions) < 5:
            print("⚠ Less than 5 questions generated. Using fallback questions.")
            questions = fallback_questions(resume_text, job_description, github_repos)

        return questions

    except Exception as e:
        print("❌ Error in question generation:", str(e))
        return fallback_questions(resume_text, job_description, github_repos)


def fallback_questions(resume_text, job_description, github_repos):
    return [
        "Tell me about yourself?",
        "What specific technical skills have you used in your past experience?",
        "How does your GitHub project align with the job requirements?",
        "Which programming language are you most comfortable with and why?",
        "Can you walk through one of your past work experiences?"
    ]


def evaluate_answer_similarity(question, answer):
    if not answer or answer.strip().lower() == "no answer provided.":
        return 0.0  # Treat blank responses as 0

    embeddings = similarity_model.encode([question, answer])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(similarity * 100, 2)

def generate_questions_with_llama_fallback(resume_text, job_description, github_repos):
    # Simpler fallback prompt
    prompt = f"""
Create 5 technical interview questions.

1 and 2 from this resume:
{resume_text}

3 to 5 from this job description and GitHub repos:
Job: {job_description}
Repos: {', '.join([repo['name'] for repo in github_repos])}

Numbered list only.
"""

    input_ids = llama_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024)
    output = llama_model.generate(
        **input_ids,
        max_new_tokens=250,
        num_return_sequences=1,
        num_beams=4,
        early_stopping=True,
    )
    result = llama_tokenizer.decode(output[0], skip_special_tokens=True)

    questions = []
    for line in result.strip().split("\n"):
        match = re.match(r"^\d+[\).\-:]?\s*(.+)", line.strip())
        if match:
            questions.append(match.group(1).strip())

    return questions[:5]


def ask_question_and_score(question, ideal_answer):
    speak(question)
    user_answer = listen()
    return compute_similarity(user_answer, ideal_answer) if user_answer else 0.0

# ------------------ MAIN FLOW ------------------
def process_resume(resume_path, job_description):
    resume_text = extract_text_from_pdf(resume_path)
    clickable = extract_clickable_links(resume_path)
    
    github_profile_url = clickable["github_profile"]
    github_repo_url = clickable["github_repo"]
    linkedin_url = clickable["linkedin"]

    if not github_profile_url and github_repo_url:
        github_profile_url = f"https://github.com/{github_repo_url.split('/')[-2]}"

    github_repos = fetch_github_repos(github_profile_url) if github_profile_url else []

    ats_resume_score = compute_similarity(resume_text, job_description)
    project_name_match_score = compute_project_name_match_score(resume_text, github_repos)
    language_score = compute_language_match_score(resume_text, job_description, github_repos)

    pre_interview_score = round((ats_resume_score + project_name_match_score + language_score) / 3, 2)

    interview_score = 0
    if pre_interview_score >= 30:
        print("\n🧠 Generating Interview Questions...\n")
        questions = generate_questions_with_llama(resume_text, job_description, github_repos)
        sample_ideal_answer = "A relevant and accurate answer to the question."  # Placeholder
        scores = []
        for i, q in enumerate(questions, 1):
            print(f"\n📢 Question {i}: {q}")
            score = ask_question_and_score(q, sample_ideal_answer)
            print(f"✅ Similarity Score: {score}/100")
            scores.append(score)
        interview_score += round(sum(scores) / len(scores), 2)
    else:
        print("❌ ATS score is too low. Interview not conducted.")

    final_ats_score = round((ats_resume_score + language_score + project_name_match_score + interview_score) / 4, 2)

    
    # 📊 Final Score Summary
    print("\n------ 📊 Score Summary ------")
    print(f"📄 ATS Resume Score: {ats_resume_score}/100")
    print(f"📁 Project Name Match Score: {project_name_match_score}/100")
    print(f"🧠 Language Match Score: {language_score}/100")
    print(f"🎤 Interview Score: {interview_score}/100")
    print(f"\n🔥 Final ATS Score: {final_ats_score}/100")

# ------------------ RUN ------------------
resume_path = r"E:\Shivanshu\shivanshu-resume.pdf"
job_description = "Looking for a web developer skilled in JavaScript, Python, and React."
process_resume(resume_path, job_description)
