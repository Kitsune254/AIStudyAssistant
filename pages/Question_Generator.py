import os
import re
import json
import time
import random

import fitz  # PyMuPDF
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# -------------------- Setup --------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    st.error("GEMINI_API_KEY is not set. Please add it to your environment or .env file.")
    st.stop()
genai.configure(api_key=gemini_api_key)

st.set_page_config(
    page_title="Thinkr",
    page_icon="📖"
)

st.title("PDF Question Generator & Evaluator")

st.write("Generate sample questions from a PDF to test your knowledge of the study material. Answer the questions and get the scores"
         " and explanation to your answers to know how much you have understood")

# -------------------- PDF utils --------------------
def extract_text_from_pdf(file) -> str:
    """Extract full text from a PDF file-like object using PyMuPDF."""
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = []
    for page in doc:
        text.append(page.get_text())
    return "\n".join(text).strip()

# -------------------- LLM helpers --------------------
def clean_json_string(json_str: str) -> str:
    """Remove common formatting issues from LLM output to improve JSON parsing."""
    # Strip code fences
    json_str = re.sub(r"```(?:json)?", "", json_str).strip("` \n\t\r")
    # Remove trailing commas before ] or }
    json_str = re.sub(r",\s*(\]|\})", r"\1", json_str)
    return json_str.strip()

def generate_questions_from_text(text: str, n: int = 5):
    """
    Ask Gemini to create questions in strict JSON. Shows a progress bar while generating.
    Returns a list of question dicts with fields:
    - type: "mcq" or "open"
    - question: str
    - options: list[str] (for mcq)
    - answer: str (correct option for mcq, expected answer for open)
    """
    progress = st.progress(0)
    status = st.empty()

    # Stage 1: starting
    status.text("Starting question generation...")
    progress.progress(10)
    time.sleep(0.2)

    # (Optional) Trim very large texts to keep prompt manageable
    MAX_CHARS = 30000
    trimmed_text = text[:MAX_CHARS]

    # Stage 2: sending to Gemini
    status.text("Sending prompt to Gemini 2.5 Flash...")
    progress.progress(35)

    model = genai.GenerativeModel("gemini-2.5-flash")
    prompt = f"""
You are a JSON-only generator.
Create exactly {n} questions from the given text.
Randomly choose each question to be either:
- Multiple-choice (type "mcq") with 4 options and exactly 1 correct answer,
- Open-ended (type "open") requiring a short written response.

Rules:
- Respond with VALID JSON ONLY. No commentary, no markdown, no trailing commas.
- For MCQs, "answer" must be EXACTLY one of the strings in "options".
- Keep questions concise and specific to the text.

Output JSON schema (array of length {n}):
[
  {{
    "type": "mcq",
    "question": "Question text",
    "options": ["Option 1","Option 2","Option 3","Option 4"],
    "answer": "Option 2"
  }},
  {{
    "type": "open",
    "question": "Question text",
    "answer": "Expected short answer"
  }}
]

Text:
{trimmed_text}
    """.strip()

    # Stage 3: model call
    response = model.generate_content(prompt)
    raw_output = (response.text or "").strip()
    progress.progress(55)
    status.text("Extracting and cleaning JSON...")

    # Stage 4: extract JSON array if extra text slipped in
    match = re.search(r"\[.*\]", raw_output, re.DOTALL)
    if match:
        raw_output = match.group(0)

    raw_output = clean_json_string(raw_output)

    # Stage 5: parse JSON
    try:
        questions = json.loads(raw_output)
        # Validate & normalize
        normalized = []
        for q in questions:
            qtype = q.get("type", "").strip().lower()
            qtext = q.get("question", "").strip()
            ans = q.get("answer", "").strip()
            if qtype == "mcq":
                opts = [str(o).strip() for o in q.get("options", []) if str(o).strip()]
                # If AI messed up and answer not in options, add it then dedup
                if ans and ans not in opts:
                    opts.append(ans)
                # Keep only first 4 options
                opts = opts[:4] if len(opts) >= 4 else opts
                # Shuffle options for display
                random.shuffle(opts)
                normalized.append({
                    "type": "mcq",
                    "question": qtext,
                    "options": opts,
                    "answer": ans  # keep the CORRECT option text for evaluation later
                })
            else:
                normalized.append({
                    "type": "open",
                    "question": qtext,
                    "answer": ans
                })
        progress.progress(100)
        status.text("✅ Questions generated successfully!")
        time.sleep(0.3)
        status.empty()
        return normalized
    except json.JSONDecodeError as e:
        status.empty()
        progress.progress(0)
        st.error(f"❌ Could not parse questions JSON: {e}")
        with st.expander("Show raw AI output"):
            st.code(raw_output)
        return []

def evaluate_open_answers_with_ai(pdf_text: str, open_qas: list[dict]) -> list[str]:
    """
    Uses Gemini to give brief feedback for open-ended answers.
    Each item in open_qas: {"question":..., "expected":..., "user":...}
    Returns a list of short feedback strings.
    """
    if not open_qas:
        return []
    model = genai.GenerativeModel("gemini-2.5-flash")
    # Keep context concise
    context = pdf_text[:15000]

    items = []
    for i, qa in enumerate(open_qas, start=1):
        items.append(
            f"{i}. Q: {qa['question']}\n"
            f"   Expected: {qa['expected']}\n"
            f"   User: {qa['user']}\n"
        )
    joined = "\n".join(items)

    prompt = f"""
You are grading short answers based on the provided source text.
For each item, give a brief verdict "Correct", "Partially correct", or "Incorrect" with ONE short sentence of feedback.
Return plain text, one line per item in the same order, prefixed with the item number.

Source text (excerpt):
{context}

Items:
{joined}
""".strip()

    resp = model.generate_content(prompt)
    feedback = (resp.text or "").strip().splitlines()
    # Ensure we return same length (fallbacks)
    if len(feedback) < len(open_qas):
        feedback += ["(No feedback generated)"] * (len(open_qas) - len(feedback))
    return [line.strip() for line in feedback]

# -------------------- Session State --------------------
if "questions" not in st.session_state:
    st.session_state.questions = []        # list of dicts
if "answers" not in st.session_state:
    st.session_state.answers = {}          # idx -> user answer
if "pdf_text" not in st.session_state:
    st.session_state.pdf_text = ""         # full text for evaluation context

# -------------------- UI --------------------
uploaded_pdf = st.file_uploader("Upload a PDF", type=["pdf"])

num_questions = st.slider("Number of questions to generate", 1, 10, 5)

if uploaded_pdf and st.button("Generate Questions"):
    # Extract text
    with st.spinner("Extracting text from PDF..."):
        pdf_text = extract_text_from_pdf(uploaded_pdf)
    if not pdf_text:
        st.error("No text could be extracted from this PDF.")
    else:
        st.session_state.pdf_text = pdf_text
        # Generate questions (with progress bar)
        qs = generate_questions_from_text(pdf_text, n=num_questions)
        st.session_state.questions = qs
        st.session_state.answers = {}  # reset any previous answers

# Render questions
if st.session_state.questions:
    st.subheader("📝 Questions")
    for i, q in enumerate(st.session_state.questions):
        st.markdown(f"**Q{i+1}. {q['question']}**")

        if q["type"] == "mcq":
            # Radio buttons (choices displayed below the question)
            choice = st.radio(
                "Select your answer:",
                q["options"],
                key=f"mcq_{i}",
                index=None
            )
            st.session_state.answers[i] = choice if choice is not None else ""
        else:
            # Open-ended text area below the question
            ans = st.text_area(
                "Your answer:",
                key=f"open_{i}",
                height=80,
                placeholder="Type your response here..."
            )
            st.session_state.answers[i] = ans.strip()

    if st.button("Evaluate Answers"):
        st.subheader("📊 Evaluation")
        total = len(st.session_state.questions)
        mcq_correct = 0
        open_items_for_ai = []

        for i, q in enumerate(st.session_state.questions):
            user_ans = st.session_state.answers.get(i, "")
            correct = q.get("answer", "")

            if q["type"] == "mcq":
                if user_ans and user_ans.strip() == correct.strip():
                    mcq_correct += 1
                    st.success(f"Q{i+1}: ✅ Correct\n\nYour answer: {user_ans}")
                else:
                    st.error(f"Q{i+1}: ❌ Incorrect\n\nYour answer: {user_ans or '(no answer)'}\n\n**Correct:** {correct}")
            else:
                # Defer grading to AI for open-ended; still show expected
                st.info(f"Q{i+1}: Your answer: {user_ans or '(no answer)'}\n\n**Expected:** {correct}")
                open_items_for_ai.append({
                    "question": q["question"],
                    "expected": correct,
                    "user": user_ans or "(no answer)"
                })

        # Summaries
        st.write("---")
        st.write(f"**MCQ score:** {mcq_correct} / {sum(1 for q in st.session_state.questions if q['type']=='mcq')}")

        # Optional AI feedback for open-ended
        if open_items_for_ai:
            with st.spinner("Getting brief feedback for open-ended answers..."):
                feedback_lines = evaluate_open_answers_with_ai(st.session_state.pdf_text, open_items_for_ai)
            st.subheader("🧠 Open-ended Feedback")
            for idx, line in enumerate(feedback_lines, start=1):
                st.write(f"{idx}. {line}")
