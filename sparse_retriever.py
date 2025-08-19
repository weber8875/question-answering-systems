from langchain.retrievers import BM25Retriever

def build_sparse_retriever(docs, k: int = 4):
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = k
    return retriever
