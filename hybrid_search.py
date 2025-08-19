from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever
from langchain.schema.document import Document

def build_hybrid_retriever(docs, persist_path: str = None, k: int = 4):

    embedding_model = HuggingFaceEmbeddings(model_name="facebook/contriever")
    vectordb = FAISS.from_documents(docs, embedding_model)

    # 如果有提供儲存路徑，就儲存
    if persist_path:
        vectordb.save_local(persist_path)

    dense_retriever = vectordb.as_retriever(search_kwargs={"k": k})

    sparse_retriever = BM25Retriever.from_documents(docs)
    sparse_retriever.k = k
    

    def hybrid_get_relevant_documents(query: str):
        dense_results  = dense_retriever.get_relevant_documents(query)
        sparse_results  = sparse_retriever.get_relevant_documents(query)
        
        combined = {doc.page_content + str(doc.metadata):doc for doc in dense_results  + sparse_results}
        return list(combined.values())

    class HybridRetriever:
        def get_relevant_documents(self, query:str):
            return hybrid_get_relevant_documents(query)

    return HybridRetriever()