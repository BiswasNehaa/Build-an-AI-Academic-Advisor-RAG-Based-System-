#understand user query


import re
from app.utils import load_careers, load_courses

def parse_query(query: str):
    query_lower = query.lower()

    careers = load_careers()
    courses = load_courses()

    detected_career = None
    completed_courses = []

    # Detect career using direct matching from dataset
    for career_key in careers.keys():
        readable_name = career_key.replace("_", " ")
        if readable_name in query_lower:
            detected_career = career_key
            break

    # Extract valid course IDs from dataset
    valid_course_ids = {course["course_id"] for course in courses}

    # Extract tokens like CS101, EC202 etc.
    tokens = re.findall(r'[A-Z]{2,}\d{2,3}', query.upper())

    for token in tokens:
        if token in valid_course_ids:
            completed_courses.append(token)

    completed_courses = list(set(completed_courses))

    return {
        "career": detected_career,
        "completed_courses": completed_courses
    }