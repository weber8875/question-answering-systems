from langchain.text_splitter import RecursiveCharacterTextSplitter

def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=600,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "。", "！", "？"]
        )
    
    return splitter.split_documents(docs)
