import os
import json
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

def create_vector_db():
    json_path='../Data/courses.json'
    if not os.path.exists(json_path):
        print(f"Error : {json_path} not found")
        return
    
    with open (json_path,'r')  as f:
        courses=json.load(f)
        
    documents=[]
    for course in courses:
        # get metadata
        outcomes_text =" ".join(course.get('course_outcome',[]))
        topics_text=" ".join(course.get('topics',[]))
        
        # This is what the AI "reads" to find matches
        page_content = (
            f"Course: {course['name']}. "
            f"Description: {course['description']}. "
            f"Topics covered: {topics_text}. "
            f"Learning Outcomes: {outcomes_text}"
        )
        
        metadata = {
            "course_id": course['course_code'],
            "name":course['name'],
            "prerequisites": course.get('prerequisites',[]),
            "credits": course.get('credits',0),
            "outcomes":outcomes_text
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
    