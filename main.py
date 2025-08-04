from loaders.pdf_loader import load_pdf
from chunkers.basic_splitter import split_documents
from retrievers.dense_retriever import build_dense_retriever
from generators.gemini_generator import generate_answer

docs = load_pdf("data/g6-final.pdf")
chunks = split_documents(docs)
print(f"🔹 共產生 {len(chunks)} 個 chunks")
print(f"📏 平均長度：{sum(len(c.page_content) for c in chunks)//len(chunks)}")


retriever = build_dense_retriever(chunks)

query = "EPA 對 SOP 編寫語氣或格式有什麼具體建議？？"
docs = retriever.invoke(query)
context = "\n".join([doc.page_content for doc in docs])

print("💡 回答：", generate_answer(context, query))

