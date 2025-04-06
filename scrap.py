from fastapi import FastAPI, HTTPException
from openai import OpenAI
import os
from typing import List
from pymongo import MongoClient
import os
from sentence_transformers import SentenceTransformer
import numpy as np
import PyPDF2
from typing import List

# Connect to MongoDB
client = MongoClient(os.getenv("MONGODB_URI"))
db = client["rag_system"]

# Collections for RAG
documents = db["documents"]  # Store your source documents
embeddings = db["embeddings"]  # Store document embeddings
conversations = db["conversations"]  # Store chat history

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def store_document(text, metadata=None):
    """Store document and its embedding in MongoDB"""
    embedding = embedding_model.encode(text).tolist()
    
    doc_id = documents.insert_one({
        "text": text,
        "metadata": metadata or {}
    }).inserted_id
    
    embeddings.insert_one({
        "doc_id": doc_id,
        "embedding": embedding,
        "text": text[:200]  # Store snippet for reference
    })
    
    return doc_id

def search_documents(query, top_k=3):
    """Search for relevant documents using vector similarity"""
    query_embedding = embedding_model.encode(query).tolist()
    
    # MongoDB Atlas Search query (if using Atlas)
    # Alternatively, use approximate nearest neighbor search
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": query_embedding,
                "numCandidates": 50,
                "limit": top_k
            }
        },
        {
            "$project": {
                "text": 1,
                "doc_id": 1,
                "score": {"$meta": "vectorSearchScore"}
            }
        }
    ]
    
    results = list(embeddings.aggregate(pipeline))
    
    # Get full document texts
    doc_ids = [r["doc_id"] for r in results]
    docs = list(documents.find({"_id": {"$in": doc_ids}}))
    
    return [{"text": d["text"], "score": r["score"]} for d, r in zip(docs, results)]


app = FastAPI()
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_response_with_rag(query: str, context: List[str]) -> str:
    """Generate response using LLM with RAG context"""
    context_str = "\n\n".join([f"Reference {i+1}:\n{text}" for i, text in enumerate(context)])
    
    prompt = f"""You are a helpful assistant. Use the following context to answer the question.
    
Context:
{context_str}

Question: {query}

Answer:"""
    
    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7
    )
    
    return response.choices[0].message.content

@app.post("/api/rag-chat")
async def rag_chat_endpoint(query: str, conversation_id: str = None):
    # 1. Retrieve relevant documents
    relevant_docs = search_documents(query)
    contexts = [doc["text"] for doc in relevant_docs]
    
    # 2. Generate response with RAG
    response = generate_response_with_rag(query, contexts)
    
    # 3. Store conversation
    if conversation_id:
        conversations.update_one(
            {"_id": conversation_id},
            {"$push": {
                "messages": {
                    "query": query,
                    "response": response,
                    "contexts": contexts,
                    "timestamp": datetime.now()
                }
            }}
        )
    else:
        conversation_id = conversations.insert_one({
            "messages": [{
                "query": query,
                "response": response,
                "contexts": contexts,
                "timestamp": datetime.now()
            }]
        }).inserted_id
    
    return {
        "response": response,
        "conversation_id": str(conversation_id),
        "references": [{"text": doc["text"][:200], "score": doc["score"]} for doc in relevant_docs]
    }

def ingest_pdf(file_path: str, metadata: dict = None) -> List[str]:
    """Extract text from PDF and store in MongoDB"""
    doc_ids = []
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            text = reader.pages[page_num].extract_text()
            if text.strip():
                doc_id = store_document(text, {
                    **metadata,
                    "source": file_path,
                    "page": page_num + 1
                })
                doc_ids.append(doc_id)
    return doc_ids

# Example usage
ingest_pdf("knowledge_base.pdf", {"type": "manual", "department": "HR"})
 