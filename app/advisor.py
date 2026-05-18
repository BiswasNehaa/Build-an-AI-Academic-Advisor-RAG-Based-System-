import os
import re
import json
from dotenv import load_dotenv
from fuzzywuzzy import fuzz  # Requires: pip install fuzzywuzzy python-Levenshtein
from langchain_groq import ChatGroq
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv() 

# --- DATA LOADING ---

# Load career data at startup
try:
    with open('../Data/career.json', 'r') as f:
        career_data = json.load(f)
except FileNotFoundError:
    print("Warning: career.json not found. Career matching will be disabled.")
    career_data = {}

# --- HELPER FUNCTIONS ---

def get_career_keywords(career_goal):
    """Fuzzy match user's career goal against career.json keys."""
    career_goal_clean = career_goal.lower().strip()
    best_match = None
    best_score = 0
    
    for career_key in career_data.keys():
        score = fuzz.partial_ratio(career_goal_clean, career_key.lower())
        if score > best_score:
            best_score = score
            best_match = career_key
    
    if best_score >= 60 and best_match:
        print(f"[Career Matched] '{career_goal}' → '{best_match}' (score: {best_score})")
        return career_data[best_match]['required_skills']
    
    print(f"[Career] No direct match for '{career_goal}', LLM will infer path.")
    return []

def is_course_code_query(query):
    """Detects if the user is asking about a specific code (e.g., BCS401)."""
    return bool(re.match(r'^[A-Z]{2,5}\d{3}[A-Z]?$', query.strip().upper()))

def clean_subject_input(user_input):
    """Splits input by common delimiters and cleans whitespace."""
    raw_list = re.split(r'[,;/\n]+', user_input)
    return [sub.strip().upper() for sub in raw_list if sub.strip()]

def is_course_satisfied(completed_upper_list, target_string):
    """Checks if any completed course satisfies a prerequisite or current course."""
    target_upper = target_string.upper()
    for completed in completed_upper_list:
        if completed in target_upper or target_upper in completed:
            return True
        codes_in_target = re.findall(r'[A-Z]{2,6}\d{3}[A-Z0-9/]*', target_upper)
        for code in codes_in_target:
            if completed in code or code in completed:
                return True
        if fuzz.partial_ratio(completed, target_upper) >= 75:
            return True
    return False

def enrich_completed_list(completed_names, vector_db):
    """Finds course codes for names provided by user to improve matching accuracy."""
    enriched = list(completed_names)
    for name in completed_names:
        results = vector_db.similarity_search(name, k=1)
        if results:
            meta = results[0].metadata
            c_code = meta.get('course_id', '').upper()
            c_name = meta.get('name', '').upper()
            if fuzz.partial_ratio(name, c_name) >= 80:
                if c_code and c_code not in enriched:
                    enriched.append(c_code)
    return enriched

# --- CORE MODEL SETUP ---

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
# Ensure faiss_index path is correct for your directory structure
vector_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# --- ADVISOR LOGIC ---

def academic_advisor(query, completed_courses, career_goal, student_credit):
    career_keywords = get_career_keywords(career_goal)
    
    if is_course_code_query(query):
        all_docs = vector_db.similarity_search(query.upper(), k=5)
        exact = [d for d in all_docs if d.metadata.get('course_id','').upper() == query.strip().upper()]
        docs = exact if exact else all_docs
    else:
        docs = vector_db.similarity_search(query, k=25)
    
    eligible_pool = []
    excluded = []
    
    for doc in docs:
        course_id = doc.metadata.get('course_id','').upper()
        course_name = doc.metadata.get('name','').upper()
        
        if is_course_satisfied(completed_courses, f"{course_id} {course_name}"):
            continue
        
        prereqs = doc.metadata.get('prerequisites', [])
        missing = [p for p in prereqs if not is_course_satisfied(completed_courses, p)]
        
        if not missing:
            eligible_pool.append({
                "course_id": course_id,
                "course_name": doc.metadata.get('name', "Unknown"),
                "credits": doc.metadata.get('credits', 0),
                "reason": doc.page_content[:200]
            })
        else:
            excluded.append({
                "course": course_name, 
                "missing_prereq": ", ".join(missing)
            })
            
    system_prompt = f"""
    You are a Senior Academic Mentor. 
    Student Career Goal: {career_goal}
    Required Skills for Goal: {career_keywords if career_keywords else "General Computer Science and AI logic."}

    Return ONLY a valid JSON object. No preamble.
    SCHEMA:
    {{
      "message": "One encouraging sentence.",
      "enroll_now": [
        {{"course_id": "...", "course_name": "...", "credits": 0, "why": "..."}}
      ],
      "unlock_next": [
        {{"complete_first": "...", "this_will_unlock": "..."}}
      ]
    }}

    Rules:
    - Prioritize core technical courses (Math, Algorithms, Python) over general electives.
    - Max 3 most relevant recommendations in 'enroll_now'.
    """
    
    prompt_input = {
        "eligible_pool": eligible_pool[:5],
        "excluded_from_logic": excluded[:3]
    }

    response = llm.invoke([
        ("system", system_prompt),
        ("user", f"Analyze and process: {json.dumps(prompt_input)}")
    ])
    return response.content

# --- MAIN EXECUTION ---

if __name__ == "__main__":
    print("--- Senior Academic Advisor (Fuzzy & Career Edition) ---")
    goal = input("Career Goal: ")
    comp_input = input("Completed Courses: ")
    
    raw_completed = clean_subject_input(comp_input)
    print("Enriching course history...")
    completed = enrich_completed_list(raw_completed, vector_db)
    print(f"Final History List: {completed}")

    while True:
        query = input("\nQuery (or 'exit'): ")
        if query.lower() in ['exit', 'quit']: break
        
        print("Analyzing academic path...")
        response = academic_advisor(query, completed, goal, 16)

        try:
            data = json.loads(response)
            print(f"\n💬 {data['message']}\n")
            
            if data.get('enroll_now'):
                print("✅ RECOMMENDED TO ENROLL:")
                for c in data['enroll_now']:
                    print(f"  [{c['course_id']}] {c['course_name']} ({c.get('credits', 'N/A')} cr) - {c.get('why', '')}")
            
            if data.get('unlock_next'):
                print("\n🗺️  NEXT MILESTONES:")
                for s in data['unlock_next']:
                    print(f"  Finish: {s['complete_first']} -> To Unlock: {s['this_will_unlock']}")

        except Exception:
            print(f"Raw Output: {response}")