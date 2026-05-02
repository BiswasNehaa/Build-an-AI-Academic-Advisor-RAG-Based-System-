'''from app.utils import load_courses, load_careers

courses = load_courses()
careers = load_careers()

print("Courses loaded:", len(courses))
print("First course:", courses[0])

print("\nCareers loaded:", len(careers))
print("Careers keys:", careers.keys()) '''


"""from app.parser import parse_query


queries = [
    "I want to become a data scientist and I completed CS101 MATH101",
    "I want to be a software engineer and completed CS304",
    "Interested in web developer and done CS101 EC101",
    "I completed CS101 CS304 and want to become ai engineer"
]

for q in queries:
    print("Query:", q)
    print("Output:", parse_query(q))
    print("-" * 50) 
    
    
import langchain
import faiss
from pydantic import BaseModel

print("LangChain", langchain.__version__)
print("FAISS")
print("Pydantic")


import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
)

response = llm.invoke("Hii")
print(response.content) """