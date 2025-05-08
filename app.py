from flask import Flask, request, jsonify
from textblob import TextBlob  # لتحليل المشاعر
import requests  # للاتصال بـ Hugging Face API
import re  # لاستخراج النصوص
import os  # للوصول إلى متغيرات البيئة
from flask_cors import CORS  # للسماح بطلبات من الواجهات الأمامية
from dotenv import load_dotenv
load_dotenv()
# تهيئة التطبيق
app = Flask(__name__)
CORS(app)  # تمكين CORS للجميع (*)

# إعدادات Hugging Face API
HF_API_URL = "https://api-inference.huggingface.co/models/bigscience/bloomz"
HF_TOKEN = os.getenv("HF_TOKEN")  # مفتاح API من إعدادات Vercel

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---- دوال مساعدة ----
def generate_text(prompt):
    """إرسال طلب إلى Hugging Face API لتوليد النص"""
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={
                "inputs": prompt,
                "parameters": {"max_length": 300}  # للتحكم في طول الإجابة
            },
            timeout=15  # حد أقصى للانتظار (ثانية)
        )
        if response.status_code == 200:
            return response.json()[0]["generated_text"]
        else:
            return f"Error: {response.text}"
    except Exception as e:
        return f"API Error: {str(e)}"

def extract_questions(text):
    """استخراج الأسئلة المرقمة من النص"""
    return [q.strip() for q in re.findall(r'\d+\.\s*(.*?[.?])', text)]

def analyze_sentiment(text):
    """تحليل مشاعر النص باستخدام TextBlob"""
    analysis = TextBlob(text)
    return {
        "polarity": round(analysis.sentiment.polarity, 2),  # بين -1 (سلبي) و1 (إيجابي)
        "subjectivity": round(analysis.sentiment.subjectivity, 2)  # بين 0 (موضوعي) و1 (شخصي)
    }

# ---- مسارات API ----
@app.route("/")
def home():
    return jsonify({"message": "Welcome to Interview API! Use /generate-questions or /evaluate-answer"})

@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    # الحصول على البيانات من الطلب
    data = request.get_json()
    job_role = data.get("job_role", "Software Engineer")  # قيمة افتراضية

    # إنشاء الـ prompt
    prompt = f"""
    Generate 3 technical and 2 behavioral interview questions for a {job_role} role.
    Format them as a numbered list in English.
    Example:
    1. Question one?
    2. Question two?
    """

    # توليد الأسئلة
    generated_text = generate_text(prompt)
    questions = extract_questions(generated_text)

    # إرجاع النتيجة
    return jsonify({
        "success": True,
        "questions": questions if questions else ["Could not generate questions. Try again later."]
    })

@app.route("/evaluate-answer", methods=["POST"])
def evaluate_answer():
    # التحقق من البيانات المدخلة
    data = request.get_json()
    if not data or "question" not in data or "answer" not in data:
        return jsonify({"error": "Both 'question' and 'answer' are required"}), 400

    question = data["question"]
    answer = data["answer"]

    # تحليل المشاعر
    sentiment = analyze_sentiment(answer)

    # تقييم الإجابة باستخدام الذكاء الاصطناعي
    prompt = f"""
    Evaluate this interview answer out of 10:
    Question: {question}
    Answer: {answer}
    Provide concise feedback and end with 'Rating: X/10'.
    """
    feedback = generate_text(prompt)

    # استخراج التقييم من التعليق
    rating_match = re.search(r'Rating:\s*(\d+)/10', feedback, re.IGNORECASE)
    rating = int(rating_match.group(1)) if rating_match else None

    # الإرجاع
    return jsonify({
        "success": True,
        "sentiment": sentiment,
        "feedback": feedback,
        "rating": rating
    })

# تشغيل الخادم (للتجربة المحلية فقط)
if __name__ == "__main__":
    app.run(debug=True)
