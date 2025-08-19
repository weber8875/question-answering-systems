from loaders.pdf_loader import load_pdf
from chunkers.basic_splitter import split_documents
from retrievers.hybrid_search import build_hybrid_retriever
from generators.refine_generator import generate_refined_answer
from reranker.bge_reranker import rerank

docs = load_pdf("data/g6-final.pdf")
chunks = split_documents(docs)
print(f"🔹 共產生 {len(chunks)} 個 chunks")
print(f"📏 平均長度：{sum(len(c.page_content) for c in chunks)//len(chunks)}")


retriever = build_hybrid_retriever(chunks)

query_1 = "什麼是 SOP？"
result_docs = retriever.get_relevant_documents(query_1)
print(f"\n🔎 檢索結果（for query: {query_1}）")
for i, doc in enumerate(result_docs):
    print(f"\n[Doc {i+1}] {doc.metadata.get('source', '')}\n{doc.page_content}\n")


reranked_docs = rerank(query_1, result_docs, top_n=2)
answer = generate_refined_answer(reranked_docs, query_1)

for i, doc in enumerate(reranked_docs):
    print(f"【Reranked Input {i+1}】\n{doc.page_content}\n")

print("💡 回答：", answer)

