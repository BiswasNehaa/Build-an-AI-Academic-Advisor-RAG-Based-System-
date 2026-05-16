# Main logic 
import os
import time 
from dotenv import load_dotenv
load_dotenv() 

import re
import json
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

#handle all user input
def clean_subject_input(user_input):
    raw_list=re.split(r'[,\s;]+',user_input)
    return [sub.strip().upper() for sub in raw_list if sub.strip()]

# Model Setup
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
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
        
        if course_id in completed_courses:
            continue
        
        prereqs = doc.metadata.get('prerequisites',[])
        course_credits = doc.metadata.get('credits',0)
        outcomes=doc.metadata.get('outcomes',"No specific outcomes listed")
        
        missing=[p for p in prereqs if p not in completed_courses]
        
        
        #eligible courses list
        if not missing:
            eligible_pool.append({
                "course_id": course_id,
                "course_name": doc.metadata.get('name',"Unknown"),
                "credits": course_credits,  
                "outcomes": outcomes
            })
        
        #missing pre reqs
        else:
            excluded.append({
                "course_id": course_id, 
                "reason": f"Missing prerequisite: {', '.join(missing)}"
            })
            
    # 2. LLM Selection
    system_prompt = f"""
    You are an AI Academic Advisor.
    Goal: {career_goal} | Credit Limit: {student_credit}
    
    CRITICAL RULES:
    1. Prioritize courses based on the User Goal.
    2. TOTAL CREDITS must be <= {student_credit}.
    3. For each recommendation, provide a 'reason' based ONLY on the provided 'outcomes'.
    4. Return a JSON object with this structure: {{"recommendations": [{{"id", "name", "reason", "credits"}}]}}
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
        print("---Helelo!! --")
        #print("Type 'exit' to quit anytime\n")
        
        #Student profile
        goal=input(" What is your career goal? (e.g., Data scientist):")
        completed_input=input(" Enter completed courses :")
        completed=clean_subject_input(completed_input)
        
        try:
            limit=int(input(" What is your credit limit For the semester ? (e.g., 12): "))
        except ValueError:
            print("Invalid number for credits. defaulting to 12")
            limit =12
            
        start_time=time.time()
        
        while True:
            #10 min time out
            if time.time() - start_time > 600:
                print("\n[TIMEOUT] Session expired. Goodbye!")
                break
            
            print("\n" + "-"*30)
            query = input("\nQuery (or 'forgot' to add courses / 'exit' to quit): ")
            
            if query.lower() in ["exit", "quit", "bye"]:
                print("Byeeeee !!")
                break
                
            # "forgotten" courses logic
            if query.lower() == "forgot":
                more_input = input("Enter the courses you forgot: ")
                new_courses = clean_subject_input(more_input)
                completed.extend(new_courses)
                print(f"Updated completed list: {completed}")
                continue

            print("\nFinding relevant courses...")
            try:
                response = academic_advisor(query, completed, goal, limit)
                print("\nRecommended plan:")
                print(response)
            except Exception as e:
                print(f"Error: {e}")