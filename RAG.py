from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


#Initialize model and embeddings
class RAG():
    def __init__(self, vectorStorePath):
        self.RAG_TEMPLATE = """
        You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.

        <context>
        {context}
        </context>

        Answer the following question:

        {question}"""
        self.vectorStorePath = vectorStorePath
        self.rag_prompt = ChatPromptTemplate.from_template(self.RAG_TEMPLATE)
        self.local_embeddings = OllamaEmbeddings(model="nomic-embed-text")
        self.model = ChatOllama(model="llama3.2:1b")
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
        self.chain = (
            RunnablePassthrough.assign(context=lambda input: "\n\n".join(doc.page_content for doc in input["context"]))
            | self.rag_prompt
            | self.model
            | StrOutputParser()
        )
    
    def store_embeddings(self, pdfPath):
        """
        Store the embeddings of the given PDF file into the `vectorstore` attribute.

        Args:
            pdfPath (str): Path to the PDF file to be stored.
        """

        reader = PyPDFLoader(pdfPath)
        reader = reader.load()

        all_splits = self.text_splitter.split_documents(reader)
        print("Splits done")
        vectorstore = Chroma.from_documents(documents=all_splits, embedding=self.local_embeddings, persist_directory=self.vectorStorePath)
        reader.clear()
        print("Vector DB saved")
        return

    def askQuestion(self, question):
        """
        Ask a question to the RAG model.

        Args:
            question (str): Question to be asked.

        Returns:
            tuple: A tuple containing the answer and the context retrieved from the vector store.
        """
        vectorstore = Chroma(persist_directory=self.vectorStorePath, embedding_function=self.local_embeddings)
        docs = vectorstore.similarity_search(question, k=5)
        context = ""
        for doc in docs:
            context += doc.page_content.replace("\n", " ")
            context += "\n\n"
        answer = self.chain.invoke({"context": docs, "question": question})
        return answer, context