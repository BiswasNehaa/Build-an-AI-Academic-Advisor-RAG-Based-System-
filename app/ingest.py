import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_vector_db():
    if not os.path.exists('../Data/courses.json'):
        print("Error, coutses.json not found")
        return
    
    with open ('../Data/courses.json','r')  as f:
        courses=json.load(f)
        
    documents=[]
    for course in courses:
        page_content=f"Course: {course['name']}.Description: {course['description']}"
        
        metadata = {
            "course_id": course['course_id'],
            "prerequisites": course['prerequisites'],
            "credits": course['credits'],
            "tags": course['tags']
        }
        documents.append(Document(page_content=page_content, metadata=metadata))
        
    print("Initializing embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    print("Building the vector database... ")
    vector_db = FAISS.from_documents(documents, embeddings)
    vector_db.save_local("faiss_index")
    
    print("faiss_index folder has been created.")

if __name__ == "__main__":
    create_vector_db()
    