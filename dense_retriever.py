from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def build_dense_retriever(docs, persist_path="./vectorstore"):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectordb = Chroma.from_documents(docs, embedding=embeddings, persist_directory=persist_path)
    vectordb.persist()
    return vectordb.as_retriever()
