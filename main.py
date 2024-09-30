from flask import Flask, request, jsonify
from flask_cors import CORS
from RAG import RAG
import os

UPLOAD_FOLDER = './uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app = Flask(__name__)
CORS(app)
rag = RAG("C:\\Users\\user\\Documents\\Programs\\xAI\\OllamaRAG\\vdb")

@app.route('/create_embeddings', methods=['POST'])
def create_embeddings():
    """
    Stores the vectors for for a PDF file in the vector store.
    Accepts a POST request with a single argument, "pdf_path", which is the path to the PDF file to be processed.
    Returns a JSON object with a single key, "message", indicating success or failure of the operation.
    If successful, returns a 200 status code.
    If an error occurs, returns a 400 status code and an "error" key with a descriptive error message.
    """
    print("Request received")
    data = request.get_json()
    pdf_path = data.get("pdf_path")

    if not pdf_path:
        return jsonify({"error": "No PDF path provided"}), 400
    
    try:
        print("Before calling store_embeddings")
        rag.store_embeddings(pdf_path)
        print("After calling store_embeddings")
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
        #print("Answer: ", answer, "\nContext: ", context)
        return jsonify({"answer": answer, "context": context}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        # Save the file to the upload folder
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        return jsonify({"message": f"File {file.filename} uploaded successfully"}), 200
    
if __name__ == '__main__':
    app.run(port=5000, debug=True, threaded=True)