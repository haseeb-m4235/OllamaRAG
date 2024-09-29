from flask import Flask, request, jsonify
from RAG import RAG

app = Flask(__name__)
rag = RAG()

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    """
    Stores the vectors for for a PDF file in the vector store.
    Accepts a POST request with a single argument, "pdf_path", which is the path to the PDF file to be processed.
    Returns a JSON object with a single key, "message", indicating success or failure of the operation.
    If successful, returns a 200 status code.
    If an error occurs, returns a 400 status code and an "error" key with a descriptive error message.
    """
    data = request.get_json()
    pdf_path = data.get("pdf_path")

    if not pdf_path:
        return jsonify({"error": "No PDF path provided"}), 400
    
    try:
        rag.store_embeddings(pdf_path)
        return jsonify({"message": "Embeddings created successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/ask_question', methods=['POST'])
def ask_question():
    """
    Ask a question to the RAG model.
    Accepts a POST request with a single argument, "question", which is the question to be asked.
    Returns a JSON object with two keys, "answer" and "context", which are the answer and context retrieved from the vector store.
    If successful, returns a 200 status code.
    If an error occurs, returns a 400 status code and an "error" key with a descriptive error message.
    """
    data = request.get_json()
    question = data.get("question")
    
    if not question:
        return jsonify({"error": "No question provided"}), 400

    try:
        answer, context = rag.askQuestion(question)
        return jsonify({"answer": answer, "context": context}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    app.run(port=5000, debug=True)