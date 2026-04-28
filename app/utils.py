#Load data

import json

def load_courses():
    with open("data/courses.json","r") as f:
        return json.load(f)
    
def load_careers():
    with open("data/career.json","r") as f:
        return json.load(f)
    