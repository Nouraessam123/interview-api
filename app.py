from flask import Flask, request, jsonify
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from flask_cors import CORS
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import re

app = Flask(__name__)
CORS(app)

# --- Models ---
qa_model = pipeline("text-generation", model="gpt2", device=-1)
analyzer = SentimentIntensityAnalyzer()

# --- API: Generate Questions ---
@app.route("/generate-questions", methods=["POST"])
def generate_questions():
    data = request.get_json()
    role = data.get("job_role", "Data Scientist")

    prompt = (
        f"Generate 3 behavioral and 2 technical interview questions for a {role} role. "
        "List only the questions, numbered."
    )

    output = qa_model(prompt, max_new_tokens=200, temperature=0.7)[0]["generated_text"]

    # Extract questions that contain a "?"
    questions = [line.strip() for line in output.split("\n") if "?" in line]

    return jsonify({"questions": questions})

# --- API: Submit Answer ---
@app.route("/submit-answer", methods=["POST"])
def submit_answer():
    data = request.get_json()
    question = data.get("question", "")
    answer = data.get("answer", "")

    sentiment = analyzer.polarity_scores(answer)

    feedback_prompt = (
        f"Evaluate how well the following answer responds to the interview question "
        f"in terms of relevance, completeness, and clarity.\n\n"
        f"Question: {question}\n"
        f"Answer: {answer}\n\n"
        f"Provide detailed feedback, then add a score out of 10 using this format:\n"
        f"Rating: X/10"
    )

    output = qa_model(feedback_prompt, max_new_tokens=300)[0]["generated_text"]

    rating = None
    match = re.search(r'Rating:\s*(\d+)/10', output)
    if match:
        extracted = int(match.group(1))
        if 0 <= extracted <= 10:
            rating = extracted

    return jsonify({
        "feedback": output,
        "rating": rating,
        "sentiment": sentiment
    })

# --- API: Sentiment Only ---
@app.route("/analyze-sentiment", methods=["POST"])
def analyze_sentiment():
    data = request.get_json()
    text = data.get("text", "")
    sentiment = analyzer.polarity_scores(text)
    return jsonify(sentiment)

@app.route("/", methods=["GET"])
def home():
    return "✅ Hugging Face API is running"

# --- Run App ---
if __name__ == "__main__":
    app.run(debug=True)
