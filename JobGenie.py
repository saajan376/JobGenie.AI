import streamlit as st
import nltk
import re
import requests
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
import random
import PyPDF2
from transformers import pipeline
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
import io
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
stop_words = set(stopwords.words('english'))

# Action verbs and vague phrases
ACTION_VERBS = ["developed", "led", "designed", "implemented", "optimized", "analyzed", "delivered", "created"]
VAGUE_PHRASES = ["team player", "hard worker", "responsible for", "good communication", "lots of"]

# --- Custom CSS for Beautiful UI ---
st.markdown("""
    <style>
    .chat-container {
        background-color: #f9f9f9;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stTextInput>div>input, .stTextArea>div>textarea {
        border-radius: 5px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    .stFileUploader {
        border: 2px dashed #4CAF50;
        border-radius: 5px;
        padding: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Gemini API Helper Function with Retry and Increased Timeout ---
def call_gemini_api(prompt, api_key="AIzaSyBFAZbDq0cUKULPMTcZfoiJA5WxpbIscRQ", max_retries=3, backoff_factor=0.5, timeout=30):
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    session = requests.Session()
    
    # Configure retry strategy for transient errors
    retries = Retry(
        total=max_retries,
        backoff_factor=backoff_factor,
        status_forcelist=[500, 502, 503, 504],
        allowed_methods=["POST"]
    )
    session.mount("https://", HTTPAdapter(max_retries=retries))
    
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "topK": 40, "topP": 0.95, "maxOutputTokens": 2048}
    }
    
    try:
        response = session.post(url, headers=headers, json=payload, timeout=timeout)
        response.raise_for_status()
        data = response.json()
        generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
        return generated_text.strip() if generated_text else ""
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to fetch : {e}")
        return ""

# --- Resume Analysis Function with Color Coding ---
def analyze_resume(resume_text, job_description, summarizer=None):
    job_words = word_tokenize(job_description.lower())
    resume_words = word_tokenize(resume_text.lower())
    job_keywords = {w for w in job_words if w not in stop_words and w.isalnum()}
    resume_keywords = {w for w in resume_words if w not in stop_words and w.isalnum()}
    matching_keywords = job_keywords.intersection(resume_keywords)
    missing_keywords = [kw for kw in job_keywords - resume_keywords if len(kw) > 3][:5]
    alignment_score = (len(matching_keywords) / len(job_keywords)) * 100 if job_keywords else 0
    
    # Calculate ATS Score (0-100) based on keyword match, density, and relevance
    ats_score = min(100, alignment_score * 1.2)  # Boost ATS score slightly to reflect keyword importance
    if len(resume_words) > 0 and len(job_keywords) > 0:
        keyword_density = (len(matching_keywords) / len(resume_words)) * 100
        ats_score = min(100, ats_score + (keyword_density * 0.5))  # Add density weight
    ats_score = max(0, ats_score)  # Ensure ATS score is not negative

    resume_word_freq = nltk.FreqDist(resume_words)
    key_freq = {kw: resume_word_freq[kw] for kw in matching_keywords if resume_word_freq[kw] > 0}
    skills_density = (len(resume_keywords) / len(resume_words)) * 100 if resume_words else 0
    sentences = sent_tokenize(resume_text)
    avg_sentence_length = sum(len(word_tokenize(s)) for s in sentences) / len(sentences) if sentences else 0
    total_words = len(resume_words)
    bullet_count = resume_text.count('•') + resume_text.count('-')
    vague_count = sum(resume_text.lower().count(phrase) for phrase in VAGUE_PHRASES)
    professionalism_score = min(100, (len(matching_keywords) - vague_count) * 10 + 50)

    def get_color(score):
        if score < 30:
            return "red"
        elif score < 60:
            return "yellow"
        else:
            return "green"

    suggestions = []
    if missing_keywords:
        suggestions.append("\n*Missing Skills*\nadd these:")
        suggestions.extend(f"- {kw.capitalize()} (e.g., '{random.choice(ACTION_VERBS)} {kw} solutions')" for kw in missing_keywords)
    if key_freq:
        low_freq = [k for k, v in key_freq.items() if v < 2]
        if low_freq:
            suggestions.append("\n*Keyword Frequency*\nmention more often:")
            suggestions.extend(f"- {k.capitalize()} (currently {key_freq[k]} time(s))" for k in low_freq)
    if avg_sentence_length > 20:
        suggestions.append(f"\n\n*Readability:* Avg sentence length {avg_sentence_length:.1f} words. Shorten them.")
    if total_words < 100:
        suggestions.append(f"*Length:* Only {total_words} words. Aim for 200-400.")
    elif total_words > 600:
        suggestions.append(f"*Length:* {total_words} words. Trim to 200-400.")
    if bullet_count < 3:
        suggestions.append(f"*Structure:* Only {bullet_count} bullets. Use more.")
    if vague_count > 2:
        suggestions.append(f"*Specificity:* {vague_count} vague phrases found. Be specific.")
    if not suggestions:
        suggestions.append("Your resume looks great!")

    summary = summarizer(resume_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] if summarizer else " ".join(sentences[:2])

    return {
        "summary": summary,
        "alignment_score": alignment_score,
        "alignment_color": get_color(alignment_score),
        "ats_score": ats_score,
        "ats_color": get_color(ats_score),
        "suggestions": "\n".join(suggestions),
        "keyword_count": len(matching_keywords),
        "skills_density": skills_density,
        "skills_density_color": get_color(skills_density),
        "professionalism_score": professionalism_score,
        "professionalism_color": get_color(professionalism_score)
    }

# --- Resume Optimization Function (Enhanced and Customized Version) ---
def optimize_resume(resume_text, job_description, job_title, st_session_state, summarizer=None):
    # Clean resume text to remove special characters and normalize
    resume_text = resume_text.replace('\xa0', ' ').replace('\ufe0f', '').replace('\xb7', '-').strip()
    
    # Parse resume into structured sections
    sections = {
        "Contact Information": [],
        "Professional Summary": [],
        "Skills": [],
        "Work Experience": [],
        "Education": [],
        "Certifications": [],
        "Honors-Awards": []
    }
    current_section = "Contact Information"
    
    lines = resume_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if re.match(r'^(contact|info|summary|skills|experience|education|work|projects|certifications|honors-awards)$', line.lower()):
            section_map = {
                "contact": "Contact Information",
                "info": "Contact Information",
                "summary": "Professional Summary",
                "skills": "Skills",
                "experience": "Work Experience",
                "work": "Work Experience",
                "projects": "Work Experience",
                "education": "Education",
                "certifications": "Certifications",
                "honors-awards": "Honors-Awards"
            }
            current_section = section_map.get(line.lower(), "Contact Information")
            if current_section not in sections:
                sections[current_section] = []
            continue
        if current_section in sections:
            sections[current_section].append(line)

    # Extract job keywords and skills from job description
    job_words = word_tokenize(job_description.lower())
    job_keywords = {w for w in job_words if w not in stop_words and w.isalnum() and len(w) > 3}
    resume_keywords = {w for w in word_tokenize(resume_text.lower()) if w not in stop_words and w.isalnum() and len(w) > 3}
    missing_keywords = list(job_keywords - resume_keywords)[:5]

    # Customize Professional Summary with job-specific details
    if "Professional Summary" not in sections or not sections["Professional Summary"]:
        sections["Professional Summary"] = []
    summary_skills = ", ".join(kw.capitalize() for kw in job_keywords if len(kw) > 3)[:50]
    company = st_session_state.get('company_name', 'the company')
    summary = (
        f"- Dynamic {job_title} with a passion for {summary_skills}, dedicated to driving innovative solutions at {company}. "
        f"Proven expertise in Machine Learning, C++, and problem-solving, with a track record of delivering impactful results in fast-paced environments."
    )
    sections["Professional Summary"] = [summary]

    # Customize Skills section with job-specific keywords and prioritize
    if "Skills" not in sections or not sections["Skills"]:
        sections["Skills"] = []
    current_skills = [s.replace('•', '').strip() for s in sections["Skills"] if s.startswith('•')]
    sections["Skills"] = [f"- {skill}" for skill in current_skills + [kw.capitalize() for kw in missing_keywords if kw.capitalize() not in current_skills]]

    # Customize Work Experience with job-specific achievements and action verbs
    if "Work Experience" not in sections or not sections["Work Experience"]:
        sections["Work Experience"] = []
    enhanced_experiences = []
    for exp in sections["Work Experience"]:
        if exp.startswith('-'):
            enhanced_experiences.append(exp)
        else:
            enhanced_experiences.append(f"- {exp}")
        for kw in missing_keywords:
            if kw.lower() not in exp.lower():
                enhanced_experiences.append(f"- {random.choice(ACTION_VERBS).capitalize()} {kw.capitalize()}-based initiatives, improving {job_title.lower()} performance at {company}")
    sections["Work Experience"] = enhanced_experiences

    # Ensure Contact Information includes name, job title, and professional details
    if "Contact Information" not in sections or not sections["Contact Information"]:
        sections["Contact Information"] = []
    candidate_name = sections["Contact Information"][0].strip() if sections["Contact Information"] else "Rithik Kumaran K"
    sections["Contact Information"] = [f"{candidate_name} | {job_title} | rithikkumarank@gmail.com | www.linkedin.com/in/rithikkumarank | Chennai, Tamil Nadu, India"]

    # Keep Certifications and Honors-Awards as is, but ensure formatting
    for section in ["Certifications", "Honors-Awards"]:
        if section in sections and sections[section]:
            sections[section] = [f"- {line}" if not line.startswith('-') else line for line in sections[section]]

    # Format optimized resume with professional layout
    optimized_text = "\n\n".join([f"**{section.replace(' ', ' ').upper()}**\n" + "\n".join(lines) for section, lines in sections.items()])
    
    # Truncate optimized_text to fit within token limit (e.g., 1000 characters)
    max_length = 1000  # Adjust based on model constraints
    if len(optimized_text) > max_length:
        optimized_text = optimized_text[:max_length] + "..."

    # Generate detailed summary with error handling for summarizer
    optimized_summary = ""
    if summarizer:
        try:
            optimized_summary = summarizer(optimized_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']
        except Exception as e:
            st.error(f"Failed to generate summary with summarizer: {e}")
            optimized_summary = (
                f"Resume Summary: {candidate_name} is a {job_title} with expertise in {summary_skills}, focusing on {job_title.lower()} challenges at {company}. "
                f"Highlights include strong skills in Machine Learning, C++, and problem-solving, with significant experience in tech-driven projects and leadership roles."
            )
    else:
        optimized_summary = (
            f"Resume Summary: {candidate_name} is a {job_title} with expertise in {summary_skills}, focusing on {job_title.lower()} challenges at {company}. "
            f"Highlights include strong skills in Machine Learning, C++, and problem-solving, with significant experience in tech-driven projects and leadership roles."
        )

    # Generate professional PDF
    pdf_data = generate_pdf_from_text(optimized_text)
    
    return {"optimized_text": optimized_text, "optimized_summary": optimized_summary, "original_feedback": analyze_resume(resume_text, job_description, summarizer), "pdf_data": pdf_data}

# --- Gemini API for Interview Questions with Answers ---
def generate_interview_questions(company_name, job_title, job_description):
    prompt = (
        f"Generate 8 diverse interview questions with concise, relevant sample answers for a {job_title} role at {company_name} as of February 2025. "
        f"Use the job description: '{job_description}'. "
        f"Include 3 behavioral questions based on skills in the description and 5 role-specific technical or practical questions. "
        f"Format the response as: '*Question : [question]\n*Answer:* [answer]\n*Question : [question]\n*Answer:* [answer]\n... "
        f"Ensure answers demonstrate competence, are tailored to the role, and reflect current industry standards."
    )
    response = call_gemini_api(prompt)
    
    # Handle empty or None response
    if not response:
        st.error("Failed to generate interview questions. Using fallback.")
        skills = re.findall(r'\b(?:skill|experience|tool|technology|software)\s+([a-zA-Z]+)', job_description, re.IGNORECASE)
        skills = list(set(skills))[:3] or ["problem-solving"]
        fallback = [
            f"1. Tell me about a time you demonstrated {skills[0]} at a company like {company_name}.\n   *Sample Answer:* I used {skills[0]} to improve a process, increasing efficiency by 20% at my last job.",
            f"2. How would you handle a tight deadline as a {job_title} at {company_name}?\n   *Sample Answer:* I’d prioritize tasks and use tools like Jira, as I did to meet a deadline in a past role.",
            f"3. Describe a project where you used {skills[0]} that would benefit {company_name}.\n   *Sample Answer:* I applied {skills[0]} to streamline a workflow, saving 10 hours weekly, applicable to {company_name}’s needs.",
            f"4. What’s a key skill for a {job_title}?\n   *Sample Answer:* {skills[0]} is crucial; I’ve used it to deliver projects successfully.",
            f"5. How do you approach problem-solving at {company_name}?\n   *Sample Answer:* I analyze the root cause, brainstorm solutions, and test them, like I did to resolve a critical issue.",
            f"6. Explain a technical concept relevant to {job_title}.\n   *Sample Answer:* For example, scalability in systems means designing for growth, which I’ve implemented using cloud solutions.",
            f"7. How would you improve a process at {company_name}?\n   *Sample Answer:* I'd assess current workflows and introduce automation, as I did to cut processing time by 30%.",
            f"8. Describe your experience with teamwork at {company_name}'s scale.\n   *Sample Answer:* I collaborated on a cross-functional team to launch a product, aligning with {company_name}’s collaborative culture."
        ]
        return fallback[:8]

    # Parse the response into question-answer pairs
    qa_pairs = []
    lines = response.split('\n')
    question = answer = None
    for line in lines:
        line = line.strip()
        if line.startswith("*Question"):
            if question and answer:
                qa_pairs.append(f"{len(qa_pairs) + 1}. {question}\n   *Sample Answer:* {answer}")
            question = line.replace("*Question", "").replace(":", "").strip()
            answer = None
        elif line.startswith("*Answer:*"):
            answer = line.replace("*Answer:*", "").strip()
    if question and answer:
        qa_pairs.append(f"{len(qa_pairs) + 1}. {question}\n   *Sample Answer:* {answer}")
    
    if not qa_pairs:
        st.error("No valid question-answer pairs parsed.Using fallback.")
        skills = re.findall(r'\b(?:skill|experience|tool|technology|software)\s+([a-zA-Z]+)', job_description, re.IGNORECASE)
        skills = list(set(skills))[:3] or ["problem-solving"]
        fallback = [
            f"1. Tell me about a time you demonstrated {skills[0]} at a company like {company_name}.\n   *Sample Answer:* I used {skills[0]} to improve a process, increasing efficiency by 20% at my last job.",
            f"2. How would you handle a tight deadline as a {job_title} at {company_name}?\n   *Sample Answer:* I’d prioritize tasks and use tools like Jira, as I did to meet a deadline in a past role.",
            f"3. Describe a project where you used {skills[0]} that would benefit {company_name}.\n   *Sample Answer:* I applied {skills[0]} to streamline a workflow, saving 10 hours weekly, applicable to {company_name}’s needs.",
            f"4. What’s a key skill for a {job_title}?\n   *Sample Answer:* {skills[0]} is crucial; I’ve used it to deliver projects successfully.",
            f"5. How do you approach problem-solving at {company_name}?\n   *Sample Answer:* I analyze the root cause, brainstorm solutions, and test them, like I did to resolve a critical issue.",
            f"6. Explain a technical concept relevant to {job_title}.\n   *Sample Answer:* For example, scalability in systems means designing for growth, which I’ve implemented using cloud solutions.",
            f"7. How would you improve a process at {company_name}?\n   *Sample Answer:* I’d assess current workflows and introduce automation, as I did to cut processing time by 30%.",
            f"8. Describe your experience with teamwork at {company_name}’s scale.\n   *Sample Answer:* I collaborated on a cross-functional team to launch a product, aligning with {company_name}’s collaborative culture."
        ]
        return fallback[:8]
    
    return qa_pairs[:8]  # Limit to 8 questions

# --- Gemini API for Salary Negotiation and Trends ---
def get_salary_negotiation(job_title, company_name, location=None):
    prompt = (
        f"Provide detailed salary information for a {job_title} role at {company_name} and similar top tech companies (e.g., Google, Amazon, Microsoft, Meta, Apple) as of February 2025. "
        f"Include salary ranges (low and high, in USD annual, formatted as $X,XXX,XXX - $X,XXX,XXX), 3-5 negotiation tips, and salary trends over the past 5 years (2020-2025) for this role across these companies. "
        f"Format the response exactly as: '*Salary Ranges:* [company1]: $X,XXX,XXX - $X,XXX,XXX, [company2]: $X,XXX,XXX - $X,XXX,XXX, ...\n*Negotiation Tips:* - [tip1], - [tip2], ...\n*Salary Trends (2020-2025):* [company1]: 2020: $X,XXX,XXX, 2021: $X,XXX,XXX, ..., [company2]: 2020: $X,XXX,XXX, ... "
        f"Ensure data is realistic, based on current U.S. market trends for tech roles, and professional. Use commas for thousands (e.g., $120,000)."
    )
    response = call_gemini_api(prompt)
    
    # Debug: Display raw Gemini response for troubleshooting
    st.write("🤖", response)
    
    # Robust parsing with multiple regex patterns
    salary_ranges = negotiation_tips = salary_trends = ""
    trend_data = {
        "default": {
            "2020": 65000, "2021": 68000, "2022": 72000, "2023": 75000, "2024": 78000, "2025": 81000
        }
    }  # Default trend_data to prevent UnboundLocalError
    
    if not response:
        st.error("Failed to generate salary details. Using default fallback.")
    else:
        # Try to match exact sections with flexible regex
        ranges_pattern = r'\*Salary Ranges:\*(.*?)(?=\*Negotiation Tips:|\Z)'
        tips_pattern = r'\*Negotiation Tips:\*(.*?)(?=\*Salary Trends \(2020-2025\):|\Z)'
        trends_pattern = r'\*Salary Trends \(2020-2025\):\*(.*?)(?=\Z)'
        
        ranges_match = re.search(ranges_pattern, response, re.DOTALL)
        tips_match = re.search(tips_pattern, response, re.DOTALL)
        trends_match = re.search(trends_pattern, response, re.DOTALL)
        
        # Extract and clean data
        salary_ranges = ranges_match.group(1).strip().replace('\n', ' ') if ranges_match else ""
        negotiation_tips = tips_match.group(1).strip().replace('\n', ' ').replace('- ', '') if tips_match else ""
        salary_trends = trends_match.group(1).strip().replace('\n', ' ') if trends_match else ""
        
        # Clean up formatting for better readability
        salary_ranges = re.sub(r'\s+', ' ', salary_ranges)
        negotiation_tips = "\n".join(f"- {tip.strip()}" for tip in negotiation_tips.split(',') if tip.strip())
        salary_trends = re.sub(r'\s+', ' ', salary_trends)
        
        # Try to extract trend_data from salary_trends if available
        if salary_trends:
            for company in ["Google", "Amazon", "Microsoft", "Meta", "Apple"]:
                pattern = rf"{company}: (\d{4}: \$[\d,]+, \d{4}: \$[\d,]+, \d{4}: \$[\d,]+, \d{4}: \$[\d,]+, \d{4}: \$[\d,]+, \d{4}: \$[\d,]+)"
                match = re.search(pattern, salary_trends)
                if match:
                    years = ["2020", "2021", "2022", "2023", "2024", "2025"]
                    values = [int(v.replace('$', '').replace(',', '')) for v in re.findall(r'\$[\d,]+', match.group(1))]
                    trend_data[company.lower()] = dict(zip(years, values))
    
    # Ensure trend_data is always available for Mermaid diagram
    company_key = company_name.lower()
    if company_key not in trend_data:
        company_key = "default"
    
    mermaid_code = (
        f"A[2020: ${trend_data[company_key]['2020']:,}] --> B[2021: ${trend_data[company_key]['2021']:,}]\n"
        f"B[2021: ${trend_data[company_key]['2021']:,}] --> C[2022: ${trend_data[company_key]['2022']:,}]\n"
        f"C[2022: ${trend_data[company_key]['2022']:,}] --> D[2023: ${trend_data[company_key]['2023']:,}]\n"
        f"D[2023: ${trend_data[company_key]['2023']:,}] --> E[2024: ${trend_data[company_key]['2024']:,}]\n"
        f"E[2024: ${trend_data[company_key]['2024']:,}] --> F[2025: ${trend_data[company_key]['2025']:,}]\n"
        f"[{job_title}Salary Trends]"
    )
    
    return {
        "salary_ranges": salary_ranges or "No salary ranges available from API.",
        "negotiation_tips": negotiation_tips or "- No negotiation tips available from API.",
        "salary_trends": salary_trends or "No salary trends available from API.",
        "salary_trends_mermaid": mermaid_code
    }

# --- Generate PDF from Text (Fixed Version) ---
def generate_pdf_from_text(text, filename="optimized_resume.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter, leftMargin=72, rightMargin=72, topMargin=72, bottomMargin=72)  # 1-inch margins
    styles = getSampleStyleSheet()
    
    # Define professional styles with Helvetica (more reliable font)
    name_style = ParagraphStyle(
        name='Name',
        fontSize=14,
        leading=18,
        textColor=colors.black,
        fontName='Helvetica-Bold',
        alignment=1,  # Center
        spaceAfter=15
    )
    job_title_style = ParagraphStyle(
        name='JobTitle',
        fontSize=12,
        leading=16,
        textColor=colors.grey,
        fontName='Helvetica',
        alignment=1,  # Center
        spaceAfter=20
    )
    section_heading_style = ParagraphStyle(
        name='SectionHeading',
        fontSize=12,
        leading=15,
        textColor=colors.black,
        fontName='Helvetica-Bold',
        spaceAfter=10,
        leftIndent=0
    )
    body_style = ParagraphStyle(
        name='BodyText',
        fontSize=10,
        leading=12,
        textColor=colors.black,
        fontName='Helvetica',  # Changed from Arial to Helvetica
        leftIndent=15,
        spaceAfter=5
    )
    bullet_style = ParagraphStyle(
        name='Bullet',
        fontSize=10,
        leading=12,
        leftIndent=15,
        bulletIndent=5,
        textColor=colors.black,
        fontName='Helvetica',  # Changed from Arial to Helvetica
        bulletText='•',
        spaceAfter=5
    )

    flowables = []

    # Clean text to remove special characters and normalize
    text = text.replace('\xa0', ' ').replace('\ufe0f', '').replace('\xb7', '-').strip()
    
    # Parse text to extract candidate name and job title (assumed from first lines)
    lines = text.split('\n')
    candidate_name = lines[0].strip() if lines else "Rithik Kumaran K"
    job_title = lines[1].strip() if len(lines) > 1 else "Software Engineer"

    # Add professional header with candidate name and job title
    flowables.append(Paragraph(f"{candidate_name.upper()}", name_style))
    flowables.append(Paragraph(f"{job_title.upper()}", job_title_style))

    # Add contact information if present (assumed from next lines)
    contact_info = []
    for line in lines[2:]:
        if line.strip() and not line.strip().startswith('•'):
            contact_info.append(line.strip())
        else:
            break
    if contact_info:
        contact_text = " | ".join(contact_info[:3])  # Limit to first 3 items for brevity
        flowables.append(Paragraph(contact_text, body_style))
        flowables.append(Spacer(1, 15))

    # Add sections with professional formatting
    current_section = None
    section_content = []
    section_names = ["Professional Summary", "Skills", "Work Experience", "Education", "Certifications", "Honors-Awards"]

    for line in lines:
        line = line.strip()
        if not line:
            continue
        if any(section.upper() in line for section in section_names):
            if current_section and section_content:
                flowables.append(Paragraph(current_section, section_heading_style))
                for content_line in section_content:
                    if content_line.startswith('•'):
                        flowables.append(Paragraph(content_line[2:], bullet_style))
                    else:
                        flowables.append(Paragraph(content_line, body_style))
                flowables.append(Spacer(1, 15))
            current_section = next(s for s in section_names if s.upper() in line)
            section_content = []
        elif current_section:
            section_content.append(line)

    # Add the last section if it exists
    if current_section and section_content:
        flowables.append(Paragraph(current_section, section_heading_style))
        for content_line in section_content:
            if content_line.startswith('•'):
                flowables.append(Paragraph(content_line[2:], bullet_style))
            else:
                flowables.append(Paragraph(content_line, body_style))
        flowables.append(Spacer(1, 15))

    doc.build(flowables)
    buffer.seek(0)
    return buffer.getvalue()

# --- Helper Function to Extract Text from Files ---
def extract_text_from_file(uploaded_file):
    if uploaded_file is None:
        return None
    if uploaded_file.name.endswith('.txt'):
        return uploaded_file.read().decode("utf-8")
    elif uploaded_file.name.endswith('.pdf'):
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return "".join(page.extract_text() or "" for page in pdf_reader.pages)
    return None

# --- Chatbot Logic ---
def chatbot_response(user_input, st_session_state):
    response = ""
    specific_commands = ["analyze", "optimize", "questions", "salary", "hi", "hello"]
    if "hi" in user_input.lower() or "hello" in user_input.lower():
        response = "Hey there! How can I assist you today? Try 'analyze', 'optimize', 'questions', 'salary', or ask any job-related query!"
    elif "analyze" in user_input.lower():
        if not st_session_state.get("resume_text") or not st_session_state.get("job_description"):
            response = "Please upload a resume and set job details first!"
        else:
            with st.spinner("Analyzing..."):
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                result = analyze_resume(st_session_state["resume_text"], st_session_state["job_description"], summarizer)
                response = (
                    f"*Resume Feedback:*\n\n"
                    f"*Summary:* {result['summary']}\n\n"
                    f"*Alignment:* <span style='color:{result['alignment_color']}'>{result['alignment_score']:.1f}% ({result['keyword_count']} keywords)</span>\n\n"
                    f"*ATS Score:* <span style='color:{result['ats_color']}'>{result['ats_score']:.1f}%</span>\n\n"  # Added ATS Score
                    f"*Skills Density:* <span style='color:{result['skills_density_color']}'>{result['skills_density']:.1f}%</span>\n\n"
                    f"*Professionalism:* <span style='color:{result['professionalism_color']}'>{result['professionalism_score']}%</span>\n\n"
                    f"*Suggestions:*\n{result['suggestions']}"
                )
    elif "optimize" in user_input.lower():
        if not st_session_state.get("resume_text") or not st_session_state.get("job_description") or not st_session_state.get("job_title"):
            response = "Please upload a resume, set job details, and provide a job title first!"
        else:
            with st.spinner("Optimizing..."):
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                result = optimize_resume(st_session_state["resume_text"], st_session_state["job_description"], st_session_state["job_title"], st_session_state, summarizer)
                response = (
                    f"*Optimized Resume for {st_session_state.get('job_title', 'Role')} at {st_session_state.get('company_name', 'Company')}*\n\n"
                    f"*Optimized Resume:*\n{result['optimized_text']}\n\n"
                    f"*Resume Summary:*\n{result['optimized_summary']}\n\n"
                    f"*Original Feedback:*\n{result['original_feedback']['suggestions']}"
                )
                st.download_button(
                    label="Download Optimized Resume as PDF",
                    data=result['pdf_data'],
                    file_name=f"Optimized_Resume_{st_session_state.get('job_title', 'Role')}.pdf",
                    mime="application/pdf"
                )
    elif "questions" in user_input.lower() or "interview" in user_input.lower():
        if not all([st_session_state.get("company_name"), st_session_state.get("job_title"), st_session_state.get("job_description")]):
            response = "Please set job details first!"
        else:
            with st.spinner("Generating questions with answers..."):
                questions = generate_interview_questions(st_session_state["company_name"], st_session_state["job_title"], st_session_state["job_description"])
                if questions:
                    response = f"*Interview Questions with Answers for {st_session_state['job_title']} at {st_session_state['company_name']}:*\n\n" + "\n\n".join(questions)
                else:
                    response = "Failed to generate questions. Please try again or check API key."
    elif "salary" in user_input.lower() or "negotiation" in user_input.lower():
        if not all([st_session_state.get("job_title"), st_session_state.get("company_name")]):
            response = "Please set job title and company name first!"
        else:
            with st.spinner("Estimating salary range, trends, and company comparisons..."):
                salary_info = get_salary_negotiation(st_session_state["job_title"], st_session_state["company_name"], st_session_state.get("location"))
                response = (
                    f"*Salary Negotiation for {st_session_state['job_title']} at {st_session_state['company_name']}*\n\n"
                    f"*Salary Trends Visualization (2020-2025):*\n```mermaid\n{salary_info['salary_trends_mermaid']}\n```"
                )
    elif not any(cmd in user_input.lower() for cmd in specific_commands):
        # Handle general job-related queries with Gemini API
        prompt = (
            f"Answer this job-related query professionally and concisely: '{user_input}'. "
            f"Provide insights based on current job market trends, career advice, or role-specific information for a {st_session_state.get('job_title', 'professional')} role at {st_session_state.get('company_name', 'a company')}, "
            f"if relevant. Ensure the response is clear, actionable, and formatted with bullet points or paragraphs as appropriate."
        )
        with st.spinner("Generating response..."):
            response = call_gemini_api(prompt)
            if not response:
                response = "Sorry, I couldn’t process your job-related query. Please try again or provide more details."
    else:
        response = "Try 'analyze' for feedback, 'optimize' for improvements, 'questions' for interview prep, 'salary' for negotiation tips and trends, or ask any job-related query!"
    return response

# --- Main Application ---
def main():
    st.title("JobGenie.AI: An AI-Career Assistant Chatbot")
    st.markdown("Level up your career game—boost your resume, ace interviews, and secure that bag! 💼🔥🚀")

    with st.expander("Setup Your Profile", expanded=True):
        uploaded_file = st.file_uploader("Upload Resume (txt or pdf)", type=["txt", "pdf"], key="resume_upload")
        col1, col2 = st.columns(2)
        with col1:
            company_name = st.text_input("Company Name", placeholder="e.g., Google")
            job_title = st.text_input("Job Title", placeholder="e.g., Software Engineer")
        with col2:
            job_description = st.text_area("Job Description", placeholder="Paste the job description here...", height=100)
            location = st.text_input("Location (optional)", placeholder="e.g., San Francisco, CA")
            if st.button("Save Job Details", key="save_job"):
                st.session_state.company_name = company_name
                st.session_state.job_title = job_title
                st.session_state.job_description = job_description
                st.session_state.location = location
                st.success("Job details saved!")
                st.session_state.messages.append({"role": "assistant", "content": f"Job details saved! Company: {company_name}, Title: {job_title}, Location: {location or 'Not specified'}"})

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Hi! I’m your AI career assistant. Upload your resume, set job details, and let’s get started!"}]
    if "resume_text" not in st.session_state:
        st.session_state.resume_text = ""
        st.session_state.company_name = ""
        st.session_state.job_title = ""
        st.session_state.job_description = ""
        st.session_state.location = ""

    # Handle resume upload immediately when a file is selected
    if uploaded_file and "resume_text" not in st.session_state:
        st.session_state.resume_text = extract_text_from_file(uploaded_file)
        st.session_state.messages.append({"role": "assistant", "content": "Resume uploaded successfully!"})
    elif uploaded_file and st.session_state.resume_text != extract_text_from_file(uploaded_file):
        # Update if the file changes (e.g., user uploads a new file)
        st.session_state.resume_text = extract_text_from_file(uploaded_file)
        st.session_state.messages.append({"role": "assistant", "content": "Resume updated successfully!"})

    with st.container():
        st.markdown("### Chat with Me")
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"], unsafe_allow_html=True)

        if user_input := st.chat_input("What can I help you with? (e.g., 'analyze', 'optimize', 'questions', 'salary', or any job query)"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            response = chatbot_response(user_input, st_session_state=st.session_state)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()

