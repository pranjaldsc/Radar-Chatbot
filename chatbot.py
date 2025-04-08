import json
import requests
from flask import Flask, request, jsonify, render_template, session
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Initialize Flask app with session support
app = Flask(__name__, template_folder="templates", static_folder="static")
app.secret_key = "your_secret_key"  # Required for session management

# Hugging Face API Key
HUGGING_FACE_API_KEY = "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"

# Load Q&A dataset
with open("qa_dataset.json", "r") as f:     
    qa_data = json.load(f)

# Extract questions and answers
questions = [qa["question"] for qa in qa_data]
answers = [qa["answer"] for qa in qa_data]

# Train TF-IDF Model
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

def get_answer(user_question):
    """Find the best matching answer from dataset using TF-IDF"""
    user_vector = vectorizer.transform([user_question])
    similarities = cosine_similarity(user_vector, question_vectors)

    best_match_index = similarities.argmax()
    confidence = similarities[0][best_match_index]

    if confidence > 0.5:
        return answers[best_match_index]
    return None  # No good match found

def get_hf_response(prompt):
    """Get AI-generated response from Hugging Face API"""
    url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct"
    headers = {"Authorization": f"Bearer {HUGGING_FACE_API_KEY}"}
    data = {"inputs": prompt}

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        return response.json()[0]["generated_text"]
    else:
        return "Sorry, I couldn't find an answer to your question."

@app.route("/")
def home():
    """Serve the chatbot UI"""
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat requests and detect repeated questions"""
    user_message = request.json.get("message", "").strip()

    # Check for repeated question
    last_question = session.get("last_question")
    if last_question == user_message:
        return jsonify({"response": "You've already asked this. Do you need more details?"})

    # Store current question in session
    session["last_question"] = user_message

    # Step 1: Check predefined answers
    predefined_answer = get_answer(user_message)

    if predefined_answer:
        return jsonify({"response": predefined_answer})

    # Step 2: If no predefined answer, use AI
    ai_response = get_hf_response(user_message)

    return jsonify({"response": ai_response})

if __name__ == "__main__":
    app.run(debug=True)
