from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from dotenv import load_dotenv
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import json

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
langchain_key = os.getenv("LANGCHAIN_API_KEY")
groq_base_url = "https://api.groq.com/openai/v1"

model = "llama-3.3-70b-versatile"
temperature = 0.7


llm = ChatGroq(model=model, temperature=temperature)

parser = JsonOutputParser(
    pydantic_object={
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "contact": {"type": "number"},
            "type": {"type": "array", "items": {"type": "string"}},
        },
    }
)


# load documents from a folder
def load_documents(folder_path: str) -> List[Document]:
    documents = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif filename.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
        else:
            print(f"Unsupported file type: {filename}")
            continue
        documents.extend(loader.load())
    return documents


folder_path = "docs"
documents = load_documents(folder_path)
print(f"Loaded {len(documents)} documents from the folder.")
# print(documents[0])


# split documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=200,
)
split_documents = text_splitter.split_documents(documents)
print(f"Split into {len(split_documents)} chunks.")
# print(split_documents[0])


# Embedding
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
document_embeddings = embedding_function.embed_documents(
    [split.page_content for split in split_documents]
)
# print(document_embeddings[0][:5])
print(len(document_embeddings))

# use chromadb to store embeddings
collection_name = "trail_documents"
persist_directory = "chroma_db"
vectorstore = Chroma.from_documents(
    collection_name=collection_name,
    documents=split_documents,
    embedding=embedding_function,
    persist_directory=persist_directory,
)
print("Embeddings stored in ChromaDB.")


retriver = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})


template = """
Answer the following question only based on the provided context.
Context:
{context}
Question:
{question}

"""
prompt = ChatPromptTemplate.from_template(template)


def docs2str(docs):
    return "\n\n".join([doc.page_content for doc in docs])


rag_chain = (
    {"context": retriver | docs2str, "question": RunnablePassthrough()}
    | prompt
    | llm
    | parser
)
rag_chain.invoke("What is Asteris Labs?")
