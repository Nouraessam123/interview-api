from flask import Flask, request, jsonify
from textblob import TextBlob
import requests
import re
import os
from flask_cors import CORS
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModelForCausalLM

# تهيئة التطبيق
load_dotenv()
app = Flask(__name__)
CORS(app)  # تمكين CORS للجميع

# تهيئة النموذج والرمز المميز
try:
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1", trust_remote_code=True)
except Exception as e:
    print(f"Error loading model: {str(e)}")
    model = None

HF_API_URL = "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-R1"
HF_TOKEN = os.getenv("HF_TOKEN")
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
} if HF_TOKEN else {}

def generate_text(prompt):
    """إنشاء نص باستخدام نموذج Hugging Face"""
    if not HF_TOKEN:
        return {"error": "Hugging Face token not configured"}
    
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {
                    "max_length": 100,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "do_sample": True
                }
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {str(e)}"}
    except Exception as e:
        return {"error": str(e)}

def extract_questions(text):
    """استخراج الأسئلة من النص المُولد"""
    questions = []
    if isinstance(text, list):
        text = text[0].get("generated_text", "") if text else ""
    
    # تحسين التعبير النمطي لالتقاط المزيد من أنماط الأسئلة
    patterns = [
        r'^\d+\.\s*(.*?\?)',  # أسئلة مرقمة (1. كيف...؟)
        r'^-\s*(.*?\?)',      # أسئلة مع نقاط (- كيف...؟)
        r'^\*\s*(.*?\?)',     # أسئلة مع علامات نجمية (* كيف...؟)
        r'^Q:\s*(.*?\?)',     # أسئلة مع Q:
        r'(?:^|\n)(.*?\?)'    # أي سؤال يحتوي على علامة استفهام
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, text, re.MULTILINE)
        questions.extend(match.strip() for match in matches if match.strip())
    
    return questions[:5] if questions else ["No questions generated."]

def analyze_sentiment(text):
    """تحليل مشاعر النص باستخدام TextBlob"""
    try:
        analysis = TextBlob(text)
        return {
            "polarity": round(analysis.sentiment.polarity, 2),
            "subjectivity": round(analysis.sentiment.subjectivity, 2),
            "sentiment": "positive" if analysis.sentiment.polarity > 0 else 
                        "negative" if analysis.sentiment.polarity < 0 else "neutral"
        }
    except Exception as e:
        return {"error": str(e)}

# ---- مسارات API ----
@app.route("/")
def home():
    """الصفحة الرئيسية للتحقق من عمل الخادم"""
    return jsonify({
        "message": "Welcome to the Interview API!",
        "endpoints": {
            "generate_questions": "/generate-questions (POST)",
            "evaluate_answer": "/evaluate-answer (POST)",
            "health_check": "/health (GET)"
        },
        "status": "operational"
    })

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    """توليد أسئلة مقابلة بناءً على الدور الوظيفي"""
    data = request.get_json()
    if not data or not data.get("job_role"):
        return jsonify({"error": "Missing or empty 'job_role' in request body"}), 400

    job_role = data["job_role"].strip()
    prompt = f"""
    Generate exactly 5 diverse interview questions for a {job_role} role:
    1. One technical question about required skills
    2. One behavioral question
    3. One situational question
    4. One question about problem-solving abilities
    5. One question about teamwork and collaboration
    
    Format each question on a new line starting with its number.
    """

    try:
        generated_text = generate_text(prompt)
        if "error" in generated_text:
            return jsonify({"error": generated_text["error"]}), 500

        questions = extract_questions(generated_text)
        return jsonify({
            "job_role": job_role,
            "questions": questions,
            "count": len(questions)
        })
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}"}), 500

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    """تقييم إجابة مقابلة مع تحليل المشاعر"""
    data = request.get_json()
    if not data or not data.get("question") or not data.get("answer"):
        return jsonify({"error": "Missing 'question' or 'answer' in request body"}), 400

    question = data["question"].strip()
    answer = data["answer"].strip()
    
    # تحليل المشاعر
    sentiment = analyze_sentiment(answer)
    if "error" in sentiment:
        return jsonify({"error": sentiment["error"]}), 500

    # تقييم الإجابة
    prompt = f"""
    Evaluate this interview answer on a scale from 1 to 10:
    Question: {question}
    Answer: {answer}
    
    Provide:
    - Strengths of the answer
    - Areas for improvement
    - Overall rating (as "Rating: X/10" at the end)
    """
    
    try:
        evaluation = generate_text(prompt)
        if "error" in evaluation:
            return jsonify({"error": evaluation["error"]}), 500

        # استخراج التقييم
        rating_match = re.search(r'Rating:\s*(\d+)/10', str(evaluation), re.IGNORECASE)
        rating = int(rating_match.group(1)) if rating_match else None

        return jsonify({
            "question": question,
            "answer": answer,
            "sentiment": sentiment,
            "feedback": str(evaluation),
            "rating": rating
        })
    except Exception as e:
        return jsonify({"error": f"Evaluation failed: {str(e)}"}), 500

@app.route("/health", methods=["GET"])
def health_check():
    """فحص صحة الخادم"""
    status = {
        "status": "healthy",
        "service": "Interview API",
        "version": "1.1",
        "model_loaded": bool(model),
        "huggingface_configured": bool(HF_TOKEN)
    }
    return jsonify(status)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=os.getenv("DEBUG", "False").lower() == "true")
