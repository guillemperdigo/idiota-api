# main.py
from fastapi import FastAPI
from pydantic import BaseModel
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings
from pinecone import Pinecone as PineconeClient
import os

# Configuració
os.environ["OPENAI_API_KEY"] = "sk-..."
PINECONE_API_KEY = "..."  # millor sense posar-lo a os.environ si vols mantenir-lo aquí
PINECONE_INDEX = "l-idiota"
PINECONE_ENV = "us-east-1"

# Inicia Pinecone
pc = PineconeClient(api_key=PINECONE_API_KEY)
vectorstore = Pinecone.from_existing_index(PINECONE_INDEX, OpenAIEmbeddings())

# FastAPI
app = FastAPI()

class QueryInput(BaseModel):
    question: str

@app.post("/query")
def query(input: QueryInput):
    results = vectorstore.similarity_search(input.question, k=3)
    return {"results": [doc.page_content for doc in results]}
