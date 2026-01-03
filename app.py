from flask import Flask, request, jsonify
from flask_cors import CORS

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# -----------------------
# Flask setup
# -----------------------
app = Flask(__name__)
CORS(app)

# -----------------------
# FAQ DATABASE
# -----------------------
FAQ_DATABASE = {
    "Q1": {"question": "What is Collar Check?", "answer": "Collar Check is a platform that helps professionals and employers verify career histories and hiring information in one place, reducing reliance on unverified resumes."},
    "Q2": {"question": "Who is Collar Check for?", "answer": "It is designed for both job seekers who want a credible career profile and for employers who want verified candidates which trigger faster and reliable hiring decisions."},
    "Q3": {"question": "Is Collar Check a job portal?", "answer": "It is more than a job portal. Along with job listings, it focuses on verified professional profiles and transparent employment history."},
    "Q4": {"question": "How is this different from LinkedIn or a resume?", "answer": "Resumes and social profiles are self-declared. Collar Check profiles are built around verified employment history and employer feedback."},
    "Q5": {"question": "What is a CC ID?", "answer": "A CC ID is your unique professional identity on Collar Check that stores your verified career information."},
    "Q6": {"question": "Do I need to be currently employed to join?", "answer": "No. Professionals at any career stage including freshers and those between roles can create a profile."},
    "Q7": {"question": "Who can see my profile?", "answer": "Your profile visibility is controlled by you. Employers only see what you choose to share."},
    "Q8": {"question": "Can a previous employer leave negative feedback?", "answer": "Feedback is structured and moderated to ensure fairness and relevance to professional performance."},
    "Q9": {"question": "Is my data safe?", "answer": "Yes. Collar Check follows data protection and privacy standards, and your information is never shared without consent."},
    "Q10": {"question": "How does verification work?", "answer": "Employment details are verified through employer confirmation and documented records rather than self-claims."},
    "Q11": {"question": "How long does verification take?", "answer": "Verification timelines vary depending on employer response, but the process is designed to be faster than traditional background checks."},
    "Q12": {"question": "Can I edit my verified information later?", "answer": "Yes. Updates can be requested, and changes go through verification before being reflected."},
    "Q13": {"question": "How does Collar Check help with hiring?", "answer": "It reduces hiring risk by giving access to verified career histories, saving time spent on manual background checks."},
    "Q14": {"question": "Can we still run our own background checks?", "answer": "Yes. Collar Check complements existing hiring processes; it does not replace mandatory compliance checks."},
    "Q15": {"question": "What kind of roles is this useful for?", "answer": "Any role where credibility, past experience, and performance matter, from entry-level to leadership positions."},
    "Q16": {"question": "Can we post jobs on Collar Check?", "answer": "Yes. Employers can post jobs and receive applications from verified professionals."},
    "Q17": {"question": "Is Collar Check free to use?", "answer": "Basic access is available, with advanced features offered through paid plans."},
    "Q18": {"question": "Are there charges for verification?", "answer": "Collar Check allows employees to build their profiles by getting ratings and reviews from managers/employers, which verifies their digital resume."},
    "Q19": {"question": "Will this affect my current job?", "answer": "No. Your profile is private unless you choose to make it visible or apply for roles."},
    "Q20": {"question": "What if my employer is no longer operational?", "answer": "Alternative documentation and verification methods can be used, including Digilocker integration."},
    "Q21": {"question": "Can I delete my account?", "answer": "Yes. Users can request account deletion as per privacy guidelines."},
    "T1": {"question": "I can't log in to my account. What should I do?", "answer": "Please check if you're using the registered email or mobile number. If the issue persists, use the 'Forgot Password' option or contact support."},
    "T2": {"question": "I didn't receive the OTP.", "answer": "OTPs may take a few seconds. Check spam messages or retry after 30 seconds."},
    "T3": {"question": "Can I have multiple accounts?", "answer": "Each user should maintain only one Collar Check account to ensure verification integrity."},
    "T4": {"question": "Can I change my registered email or mobile number?", "answer": "Yes. You can update contact details from your account settings."}
}

# -----------------------
# Load semantic model
# -----------------------
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

# Precompute FAQ embeddings
FAQ_IDS = list(FAQ_DATABASE.keys())
FAQ_QUESTIONS = [FAQ_DATABASE[qid]["question"] for qid in FAQ_IDS]

FAQ_EMBEDDINGS = model.encode(
    FAQ_QUESTIONS,
    convert_to_numpy=True,
    normalize_embeddings=True
)

# Similarity threshold
SIMILARITY_THRESHOLD = 0.6

# -----------------------
# Utility functions
# -----------------------
def clean_text(text: str) -> str:
    return text.strip().lower()

def embed_query(query: str):
    return model.encode([query], convert_to_numpy=True, normalize_embeddings=True)

def find_best_match(user_query: str):
    query_embedding = embed_query(clean_text(user_query))
    similarities = cosine_similarity(query_embedding, FAQ_EMBEDDINGS)[0]

    best_index = int(np.argmax(similarities))
    best_score = float(similarities[best_index])

    return FAQ_IDS[best_index], best_score

def find_similar_questions(user_query: str, top_n: int = 4):
    """Find top N similar questions to the user query"""
    query_embedding = embed_query(clean_text(user_query))
    similarities = cosine_similarity(query_embedding, FAQ_EMBEDDINGS)[0]
    
    # Get indices of top N similarities
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    similar_ids = [FAQ_IDS[idx] for idx in top_indices]
    similar_scores = [float(similarities[idx]) for idx in top_indices]
    
    return similar_ids, similar_scores

def get_answer(user_query: str):
    qid, score = find_best_match(user_query)
    
    # Get 3 similar questions to explore
    similar_ids, similar_scores = find_similar_questions(user_query, top_n=4)
    # Remove the best match from similar questions and keep only 3 others
    explore_ids = [id for id in similar_ids if id != qid][:3]

    faq = FAQ_DATABASE[qid]
    return {
        "success": True,
        "question_id": qid,
        "matched_question": faq["question"],
        "answer": faq["answer"],
        "confidence": round(score, 3),
        "low_confidence": score < SIMILARITY_THRESHOLD,
        "similar_question_ids": explore_ids
    }

# -----------------------
# Flask routes
# -----------------------
@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "online",
        "service": "Collar Check FAQ Chatbot",
        "engine": "Semantic embeddings (offline, mpnet-v2)"
    })

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    if not data or "query" not in data:
        return jsonify({"success": False, "error": "Missing query"}), 400

    query = data["query"].strip()
    if not query:
        return jsonify({"success": False, "error": "Empty query"}), 400

    return jsonify(get_answer(query))

@app.route("/api/faqs", methods=["GET"])
def get_all_faqs():
    return jsonify({
        "total": len(FAQ_DATABASE),
        "faqs": [
            {"id": qid, "question": v["question"], "answer": v["answer"]}
            for qid, v in FAQ_DATABASE.items()
        ]
    })

@app.route("/api/faq/<qid>", methods=["GET"])
def get_faq(qid):
    if qid not in FAQ_DATABASE:
        return jsonify({"success": False, "error": "FAQ not found"}), 404

    faq = FAQ_DATABASE[qid]
    return jsonify({
        "success": True,
        "question_id": qid,
        "question": faq["question"],
        "answer": faq["answer"]
    })

# -----------------------
# Run app
# -----------------------
if __name__ == "__main__":
    print("Starting Collar Check FAQ Chatbot (Offline Semantic Mode, MPNet-V2)")
    print("http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)
