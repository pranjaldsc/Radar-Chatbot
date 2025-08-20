import json
import requests
from flask import Flask, request, jsonify, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app with session support
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "your_secret_key"  # Required for session management

# Hugging Face API Key
HUGGING_FACE_API_KEY = "hf_xxxxxxx"  # replace with your key

# Load multilingual Q&A dataset
with open("qa_dataset.json", "r", encoding="utf-8") as f:
    qa_data = json.load(f)


def get_answer(user_question, lang="english"):
    """Find the best matching answer from dataset using TF-IDF in selected language"""

    # Extract questions & answers in the chosen language
    questions = [qa["question"][lang] for qa in qa_data]
    answers = [qa["answer"][lang] for qa in qa_data]

    # Train TF-IDF for this language
    vectorizer = TfidfVectorizer()
    question_vectors = vectorizer.fit_transform(questions)

    # Match user question
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarities.argmax()
    confidence = similarities[0][best_match_index]

    if confidence > 0.5:
        return answers[best_match_index], qa_data[best_match_index]["id"]  # return answer + ID
    return None, None


def get_hf_response(prompt, lang="english"):
    """Get AI-generated response from Hugging Face API"""
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    data = {"inputs": f"Answer in {lang}: {prompt}"}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Sorry, I couldn't find an answer to your question."


@app.route("/")
def home():
    """Serve the chatbot UI"""
    return render_template("index.html")


@app.route("/set_language", methods=["POST"])
def set_language():
    """Store user's language preference in session"""
    lang = request.json.get("language", "english").lower()
    if lang not in ["english", "hinglish", "hindi"]:
        lang = "english"  # default
    session["language"] = lang
    session["last_question_id"] = None  # reset previous question
    return jsonify({"message": f"Language set to {lang.title()}"})


@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests and detect repeated questions"""
    user_message = request.json.get("message", "").strip()

    # Check if language is set
    lang = session.get("language")
    if not lang:
        return jsonify({"response": "Please choose your language first: English, Hinglish, or Hindi."})

    # Step 1: Check predefined answers
    predefined_answer, qid = get_answer(user_message, lang)

    last_qid = session.get("last_question_id")

    # If same Q&A id as last time → treat as duplicate
    if qid and last_qid == qid:
        return jsonify({"response": "You've already asked this question. Do you want me to explain in more detail?"})

    # Store current question id in session
    if qid:
        session["last_question_id"] = qid
        return jsonify({"response": predefined_answer})

    # Step 2: If no predefined answer, use AI
    ai_response = get_hf_response(user_message, lang)
    session["last_question_id"] = None  # reset if AI handled it
    return jsonify({"response": ai_response})


if __name__ == "__main__":
    app.run(debug=True)
