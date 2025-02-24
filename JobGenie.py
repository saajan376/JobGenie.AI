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
from reportlab.lib.styles impor t getSampleStyleSheet
import io

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

# --- Resume Analysis Function with Color Coding ---
def analyze_resume(resume_text, job_description, summarizer=None):
    job_words = word_tokenize(job_description.lower())
    resume_words = word_tokenize(resume_text.lower())
    job_keywords = {w for w in job_words if w not in stop_words and w.isalnum()}
    resume_keywords = {w for w in resume_words if w not in stop_words and w.isalnum()}
    matching_keywords = job_keywords.intersection(resume_keywords)
    missing_keywords = [kw for kw in job_keywords - resume_keywords if len(kw) > 3][:5]
    alignment_score = (len(matching_keywords) / len(job_keywords)) * 100 if job_keywords else 0
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
        suggestions.append("*Missing Skills:* Add these:")
        suggestions.extend(f"- {kw.capitalize()} (e.g., '{random.choice(ACTION_VERBS)} {kw} solutions')" for kw in missing_keywords)
    if key_freq:
        low_freq = [k for k, v in key_freq.items() if v < 2]
        if low_freq:
            suggestions.append("*Keyword Frequency:* Mention more often:")
            suggestions.extend(f"- {k.capitalize()} (currently {key_freq[k]} time(s))" for k in low_freq)
    if avg_sentence_length > 20:
        suggestions.append(f"*Readability:* Avg sentence length {avg_sentence_length:.1f} words. Shorten them.")
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
        "suggestions": "\n".join(suggestions),
        "keyword_count": len(matching_keywords),
        "skills_density": skills_density,
        "skills_density_color": get_color(skills_density),
        "professionalism_score": professionalism_score,
        "professionalism_color": get_color(professionalism_score)
    }

# --- Resume Optimization Function (Your Provided Version) ---
def optimize_resume(resume_text, job_description, summarizer=None):
    job_keywords = {w for w in word_tokenize(job_description.lower()) if w not in stop_words and w.isalnum()}
    resume_keywords = {w for w in word_tokenize(resume_text.lower()) if w not in stop_words and w.isalnum()}
    result = analyze_resume(resume_text, job_description, summarizer)
    optimized_text = resume_text
    for kw in [kw for kw in job_keywords - resume_keywords if len(kw) > 3][:3]:
        optimized_text += f"\n• {random.choice(ACTION_VERBS).capitalize()} {kw.capitalize()}-based projects to enhance productivity."
    optimized_summary = summarizer(optimized_text, max_length=100, min_length=30, do_sample=False)[0]['summary_text'] if summarizer else optimized_text[:200]
    
    pdf_data = generate_pdf_from_text(optimized_text)
    
    return {"optimized_text": optimized_text, "optimized_summary": optimized_summary, "original_feedback": result, "pdf_data": pdf_data}

# --- Gemini API Call for Interview Questions with Answers ---
def generate_interview_questions(company_name, job_title, job_description, role=None):
    api_key = "AIzaSyBFAZbDq0cUKULPMTcZfoiJA5WxpbIscRQ"  # Replace with your actual Gemini API key
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent"
    
    # Craft prompt for Gemini API
    prompt = (
        f"Generate 8 diverse interview questions with sample answers for a {job_title} role at {company_name}. "
        f"Use the job description: '{job_description}'. "
        f"Include 3 behavioral questions based on skills in the description and 5 role-specific technical or practical questions. "
        f"Format each response as: 'Question: [question]\nAnswer: [answer]'. "
        f"Ensure answers are concise, relevant to the role, and demonstrate competence. "
        f"Do not include extra text beyond the 8 formatted question-answer pairs."
    )

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key
    }

    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0.7,
            "topK": 40,
            "topP": 0.95,
            "maxOutputTokens": 1024
        }
    }

    try:
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        data = response.json()
        generated_text = data["candidates"][0]["content"]["parts"][0]["text"]
        
        # Debug: Display raw API response
        st.write("Raw Gemini API Response:", generated_text)
        
        # Parse the generated text into questions and answers
        qa_pairs = []
        lines = generated_text.split('\n')
        for i in range(0, len(lines), 2):  # Assuming question and answer are consecutive lines
            if i + 1 < len(lines) and lines[i].startswith("Question:") and lines[i + 1].startswith("Answer:"):
                question = lines[i].replace("Question:", "").strip()
                answer = lines[i + 1].replace("Answer:", "").strip()
                qa_pairs.append(f"{len(qa_pairs) + 1}. {question}\n   *Sample Answer:* {answer}")
        
        if not qa_pairs:
            st.error("No valid question-answer pairs parsed from Gemini response.")
            return ["No questions generated due to parsing error."]
        
        return qa_pairs[:8]  # Limit to 8 questions

    except Exception as e:
        st.error(f"Failed to fetch from Gemini API: {e}. Using fallback questions.")
        # Fallback static questions with answers
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
        return fallback

# --- Salary Negotiation Function with Minimal Mermaid Diagram ---
def get_salary_negotiation(job_title, company_name, location=None):
    base_salaries = {
        "software engineer": {"low": 90000, "high": 150000},
        "marketing manager": {"low": 70000, "high": 120000},
        "nurse": {"low": 60000, "high": 100000},
        "data scientist": {"low": 95000, "high": 160000},
        "product manager": {"low": 85000, "high": 140000}
    }
    company_multipliers = {"google": 1.3, "amazon": 1.3, "microsoft": 1.25, "facebook": 1.3, "apple": 1.3}
    location_multipliers = {"san francisco": 1.2, "new york": 1.15, "seattle": 1.1, "remote": 0.95, "default": 1.0}
    
    job_key = job_title.lower()
    company_key = company_name.lower()
    location_key = location.lower() if location else "default"
    base_range = base_salaries.get(job_key, {"low": 60000, "high": 100000})
    company_factor = company_multipliers.get(company_key, 1.1)
    location_factor = location_multipliers.get(location_key, 1.0)
    low_salary = int(base_range["low"] * company_factor * location_factor)
    high_salary = int(base_range["high"] * company_factor * location_factor)
    
    trends = {
        "software engineer": {"2022": 110000, "2023": 115000, "2024": 122000, "2025": 128000, "2026": 134000},
        "marketing manager": {"2022": 82000, "2023": 86000, "2024": 90000, "2025": 93000, "2026": 96000},
        "nurse": {"2022": 70000, "2023": 73000, "2024": 76000, "2025": 79000, "2026": 82000},
        "data scientist": {"2022": 120000, "2023": 126000, "2024": 132000, "2025": 138000, "2026": 144000},
        "product manager": {"2022": 100000, "2023": 105000, "2024": 110000, "2025": 115000, "2026": 120000}
    }
    trend_data = trends.get(job_key, {"2022": 72000, "2023": 75000, "2024": 78000, "2025": 81000, "2026": 84000})
    
    mermaid_code = (
        f"[2022: ${trend_data['2022']:,}] --> [2023: ${trend_data['2023']:,}]\n"
        f"[2023: ${trend_data['2023']:,}] --> [2024: ${trend_data['2024']:,}]\n"
        f"[2024: ${trend_data['2024']:,}] --> [2025: ${trend_data['2025']:,}]\n"
        f"[2025: ${trend_data['2025']:,}] --> [2026: ${trend_data['2026']:,}]\n"
        f"[{job_title} Salary Trends]"
    )
    
    tips = [
        "Research market rates for your role using sites like Glassdoor or Payscale.",
        f"Highlight your unique skills or achievements (e.g., '{random.choice(ACTION_VERBS)} critical projects').",
        "Ask for 10-20% above their initial offer to anchor negotiations higher.",
        "Negotiate extras like signing bonuses, stock options, or remote work flexibility.",
        "Time your ask after a strong performance review or project success.",
        f"Practice your pitch: 'Based on my experience, I believe a range of ${low_salary + 5000:,}-${high_salary:,} aligns with market standards.'"
    ]
    
    return {
        "salary_range": f"${low_salary:,} - ${high_salary:,}",
        "negotiation_tips": random.sample(tips, min(4, len(tips))),
        "salary_trends_mermaid": mermaid_code
    }

# --- Generate PDF from Text (Your Provided Version) ---
def generate_pdf_from_text(text, filename="optimized_resume.pdf"):
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    style = styles["BodyText"]
    flowables = []

    for line in text.split('\n'):
        if line.strip():
            flowables.append(Paragraph(line, style))
            flowables.append(Spacer(1, 6))

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
def chatbot_response(user_input, state):
    response = ""
    if "hi" in user_input.lower() or "hello" in user_input.lower():
        response = "Hey there! How can I assist you today? Try 'analyze', 'optimize', 'questions', or 'salary'!"
    elif "analyze" in user_input.lower():
        if not state["resume_text"] or not state["job_description"]:
            response = "Please upload a resume and set job details first!"
        else:
            with st.spinner("Analyzing..."):
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                result = analyze_resume(state["resume_text"], state["job_description"], summarizer)
                response = (
                    f"*Resume Feedback:*\n\n"
                    f"*Summary:* {result['summary']}\n\n"
                    f"*Alignment:* <span style='color:{result['alignment_color']}'>{result['alignment_score']:.1f}% ({result['keyword_count']} keywords)</span>\n\n"
                    f"*Skills Density:* <span style='color:{result['skills_density_color']}'>{result['skills_density']:.1f}%</span>\n\n"
                    f"*Professionalism:* <span style='color:{result['professionalism_color']}'>{result['professionalism_score']}%</span>\n\n"
                    f"*Suggestions:*\n{result['suggestions']}"
                )
    elif "optimize" in user_input.lower():
        if not state["resume_text"] or not state["job_description"]:
            response = "Please upload a resume and set job details first!"
        else:
            with st.spinner("Optimizing..."):
                summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")
                result = optimize_resume(state["resume_text"], state["job_description"], summarizer)
                response = (
                    f"*Optimized Resume for {state['job_title']} at {state['company_name']}*\n\n"
                    f"*Summary:* {result['optimized_summary']}\n\n"
                    f"*Optimized Resume:*\n{result['optimized_text']}\n\n"
                    f"*Original Feedback:*\n{result['original_feedback']['suggestions']}"
                )
                st.download_button(
                    label="Download Optimized Resume as PDF",
                    data=result['pdf_data'],
                    file_name=f"Optimized_Resume_{state['job_title']}.pdf",
                    mime="application/pdf"
                )
    elif "questions" in user_input.lower() or "interview" in user_input.lower():
        if not all([state["company_name"], state["job_title"], state["job_description"]]):
            response = "Please set job details first!"
        else:
            with st.spinner("Generating questions with answers via Gemini API..."):
                questions = generate_interview_questions(state["company_name"], state["job_title"], state["job_description"])
                if questions:
                    response = f"*Interview Questions with Answers for {state['job_title']} at {state['company_name']}:*\n\n" + "\n\n".join(questions)
                else:
                    response = "Failed to generate questions. Please try again or check API key."
    elif "salary" in user_input.lower() or "negotiation" in user_input.lower():
        if not all([state["job_title"], state["company_name"]]):
            response = "Please set job title and company name first!"
        else:
            with st.spinner("Estimating salary range and trends..."):
                salary_info = get_salary_negotiation(state["job_title"], state["company_name"], state["location"])
                response = (
                    f"*Salary Negotiation for {state['job_title']} at {state['company_name']}*\n\n"
                    f"*Estimated Range:* {salary_info['salary_range']} (annual)\n\n"
                    f"*Negotiation Tips:*\n" + "\n".join(f"- {tip}" for tip in salary_info['negotiation_tips']) + "\n\n"
                    f"*Salary Trends (2022-2026):*\n```mermaid\n{salary_info['salary_trends_mermaid']}\n```"
                )
    else:
        response = "Try 'analyze' for feedback, 'optimize' for improvements, 'questions' for interview prep, or 'salary' for negotiation tips and trends!"
    return response

# --- Main Application ---
def main():
    st.title("JobGenie.AI:An AI-Career Assistant Chatbot")
    st.markdown("Level up your career game—boost your resume, ace interviews, and secure that bag! 💼🔥🚀")

    with st.expander("Setup Your Profile", expanded=True):
        uploaded_file = st.file_uploader("Upload Resume (txt or pdf)", type=["txt", "pdf"], help="Drop your resume here!")
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

    if uploaded_file:
        st.session_state.resume_text = extract_text_from_file(uploaded_file)
        st.session_state.messages.append({"role": "assistant", "content": "Resume uploaded successfully!"})

    with st.container():
        st.markdown("### Chat with Me")
        for message in st.session_state.messages:
            with st.chat_message(message["role"], avatar="🤖" if message["role"] == "assistant" else "👤"):
                st.markdown(message["content"], unsafe_allow_html=True)

        if user_input := st.chat_input("What can I help you with? (e.g., 'analyze', 'optimize', 'questions', 'salary')"):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_input)
            response = chatbot_response(user_input, state=st.session_state)
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(response, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()