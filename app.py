from app import Flask, request, jsonify
from RAG import RAG

app = Flask(__name__)
rag = RAG()

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    
    # Get the PDF file path from the request
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