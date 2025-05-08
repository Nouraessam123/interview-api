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
HF_API_URL = "meta-llama/Llama-2-7b-chat-hf"
HF_TOKEN = os.getenv("HF_TOKEN")  # مفتاح API من إعدادات Vercel

headers = {"Authorization": f"Bearer {HF_TOKEN}"}

# ---- دوال مساعدة ----
def generate_text(prompt):
    try:
        response = requests.post(
            HF_API_URL,
            headers=headers,
            json={"inputs": prompt, "parameters": {"max_length": 100}},  # قلل max_length لتسريع الاستجابة
            timeout=10  # قلل وقت الانتظار
        )
        response_data = response.json()
        
        # Debugging: سجّل الاستجابة كاملة
        print("Hugging Face Response:", response_data)
        
        if isinstance(response_data, list) and len(response_data) > 0:
            return response_data[0].get("generated_text", "No text generated")
        elif isinstance(response_data, dict) and "error" in response_data:
            return f"Error: {response_data['error']}"
        else:
            return f"Unexpected response: {response_data}"
    except requests.exceptions.Timeout:
        return "Error: Hugging Face API timeout"
    except Exception as e:
        return f"API Error: {str(e)}"

def extract_questions(text):
    lines = text.strip().split("\n")
    questions = []
    for line in lines:
        line = line.strip()
        match = re.match(r'^\d+\.\s*(.*)', line)
        if match:
            questions.append(match.group(1).strip())
        elif "?" in line:  # سؤال بدون رقم
            questions.append(line)
    return questions

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

    # إنشاء الـ prompt (معدّل)
    prompt = f"""
    Generate exactly 5 interview questions for a {job_role} role:
    - 3 Technical questions.
    - 2 Non-technical (behavioral or situational) questions.
    
    Format strictly as:
    1. [Technical question one]?
    2. [Technical question two]?
    3. [Technical question three]?
    4. [Non-technical question one]?
    5. [Non-technical question two]?
    """

    # توليد الأسئلة
    generated_text = generate_text(prompt)
    questions = extract_questions(generated_text)

    # إرجاع النتيجة
    return jsonify({
        "success": True,
        "questions": questions if questions else ["Could not generate questions. Try again later."]
    })
    print("Generated Text:", generated_text)

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
