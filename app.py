from flask import Flask, request, jsonify
from textblob import TextBlob  # لتحليل المشاعر
import requests  # للاتصال بـ Hugging Face API
import re  # لاستخراج النصوص
import os  # للوصول إلى متغيرات البيئة
from flask_cors import CORS  # للسماح بطلبات من الواجهات الأمامية
from dotenv import load_dotenv

# تهيئة التطبيق
load_dotenv()
app = Flask(__name__)
CORS(app)  # تمكين CORS للجميع (*)

# إعدادات Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.getenv("HF_TOKEN")  # مفتاح API من إعدادات Vercel
headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---- دوال مساعدة ----
def generate_text(prompt):
    """إرسال طلب إلى Hugging Face API لتوليد النص."""
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_length": 100}},
            timeout=10  # حد زمني للطلب
        )
        response.raise_for_status()  # رفع خطأ إذا كانت الاستجابة غير ناجحة
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": f"API Request Failed: {str(e)}"}

def extract_questions(text):
    """استخراج الأسئلة من النص المُولد باستخدام التعبير النمطي."""
    questions = []
    if isinstance(text, list):
        text = text[0].get("generated_text", "") if text else ""
    for line in text.split("\n"):
        line = line.strip()
        if re.match(r'^\d+\.\s*(.*\?)', line):  # يطابق الأسئلة المرقمة (مثل "1. كيف...؟")
            questions.append(line)
        elif "?" in line:  # يطابق أي سؤال يحتوي على "؟"
            questions.append(line)
    return questions if questions else ["No questions generated."]

def analyze_sentiment(text):
    """تحليل مشاعر النص باستخدام TextBlob."""
    analysis = TextBlob(text)
    return {
        "polarity": round(analysis.sentiment.polarity, 2),  # بين -1 (سلبي) و1 (إيجابي)
        "subjectivity": round(analysis.sentiment.subjectivity, 2)  # بين 0 (موضوعي) و1 (شخصي)
    }

# ---- مسارات API ----
@app.route("/")
def home():
    """الصفحة الرئيسية للتحقق من عمل الخادم."""
    return jsonify({
        "message": "Welcome to the Interview API!",
        "endpoints": {
            "generate_questions": "/generate-questions (POST)",
            "evaluate_answer": "/evaluate-answer (POST)",
            "health_check": "/health (GET)"
        }
    })

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    """توليد أسئلة مقابلة بناءً على الدور الوظيفي."""
    data = request.get_json()
    if not data or "job_role" not in data:
        return jsonify({"error": "Missing 'job_role' in request body"}), 400

    job_role = data["job_role"]
    prompt = f"""
    Generate exactly 5 interview questions for a {job_role} role:
    1. [Technical question one]?
    2. [Technical question two]?
    3. [Technical question three]?
    4. [Non-technical question one]?
    5. [Non-technical question two]?
    """

    generated_text = generate_text(prompt)
    if "error" in generated_text:
        return jsonify({"error": generated_text["error"]}), 500

    questions = extract_questions(generated_text)
    return jsonify({
        "job_role": job_role,
        "questions": questions
    })

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    """تقييم إجابة مقابلة مع تحليل المشاعر."""
    data = request.get_json()
    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error": "Missing 'question' or 'answer' in request body"}), 400

    question = data["question"]
    answer = data["answer"]
    sentiment = analyze_sentiment(answer)

    # تقييم الإجابة باستخدام النموذج
    prompt = f"""
    Evaluate this interview answer out of 10:
    Question: {question}
    Answer: {answer}
    Provide concise feedback and end with 'Rating: X/10'.
    """
    evaluation = generate_text(prompt)
    if "error" in evaluation:
        return jsonify({"error": evaluation["error"]}), 500

    # استخراج التقييم من النص
    rating_match = re.search(r'Rating:\s*(\d+)/10', str(evaluation), re.IGNORECASE)
    rating = int(rating_match.group(1)) if rating_match else None

    return jsonify({
        "question": question,
        "answer": answer,
        "sentiment": sentiment,
        "feedback": str(evaluation),
        "rating": rating
    })

@app.route("/health", methods=["GET"])
def health_check():
    """فحص صحة الخادم."""
    return jsonify({
        "status": "healthy",
        "service": "Interview API",
        "version": "1.0"
    })

# تشغيل التطبيق محليًا (للتطوير فقط)
if __name__ == "__main__":
    app.run(debug=True)
