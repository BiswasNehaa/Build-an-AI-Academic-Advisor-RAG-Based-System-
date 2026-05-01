# Main logic 
import os
from dotenv import load_dotenv
load_dotenv() 

import json
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

# Model Setup
embeddings=HuggingFaceBgeEmbeddings(model_name="all-MiniLM-L6-v2")
llm=ChatGroq(model="llama-3.3-70b-versatile",temperature=0)  #deterministic

#load vectordb
vector_db=FAISS.load_local("faiss_index",embeddings,allow_dangerous_deserialization=True)



def academic_advisor(query, completed_courses, career_goal, student_credit):
    # 1. Retrieval
    docs = vector_db.similarity_search(query, k=15)
    
    # 2. Setup
    eligible_pool=[]
    excluded = []
    #total_credits = 0
    
    
    for doc in docs:
        course_id = doc.metadata['course_id']
        prereqs = doc.metadata['prerequisites']
        course_credits = doc.metadata['credits']
        
        missing=[p for p in prereqs if p not in completed_courses]
        
        
        #eligible courses list
        # Eligible courses list
        if not missing:
            eligible_pool.append({
                "course_id": course_id,
                "course_name": doc.page_content.split('.')[0].replace("Course: ", ""),
                "credits": course_credits,  # <--- FIX: Use the variable name
                "tags": doc.metadata['tags']
            })
        
        #missing pre reqs
        else:
            excluded.append({
                "course_id": course_id, 
                "reason": f"Missing prerequisite: {', '.join(missing)}"
            })
            
    # LLM Selection
    system_prompt = f"""
    You are an AI Academic Advisor.
    Goal: {career_goal}
    Credit Limit: {student_credit}
    
    TASK:
    1. From the 'Eligible Pool', select the TOP 4 most relevant courses for the career goal.
    2. Ensure the TOTAL CREDITS of selected courses is <= {student_credit}.
    3. If you can't fit 4, pick as many as possible within the limit.
    4. Return ONLY a JSON object.
    """
    
    prompt_input = {
        "eligible_pool": eligible_pool,
        "excluded_from_logic": excluded[:3] # Show a few reasons for exclusion
    }

    response = llm.invoke([
        ("system", system_prompt),
        ("user", f"Process this data: {json.dumps(prompt_input)}")
    ])

    return response.content

if __name__ == "__main__":
    # Test parameters
    query = "I want to become a Data Scientist"
    completed = ["Intro to Programming"] # Change this to test prereq logic
    goal = "Data Scientist"
    limit = 12

    print("\n--- Consulting AI Academic Advisor ---")
    
    # Actually run the function!
    response = academic_advisor(query, completed, goal, limit)
    
    # Show the result
    print("\nRecommended Courses (JSON):")
    print(response)