# 🎓 AI Academic Advisor (RAG-Based System)

## 📌 Overview

The AI Academic Advisor is a Retrieval-Augmented Generation (RAG)-based system designed to assist students in making informed academic decisions. It acts as an intelligent copilot that analyzes a student’s academic history, career goals, and institutional constraints to recommend suitable courses for upcoming semesters.

Unlike traditional chatbots, this system combines **semantic retrieval** with **deterministic reasoning** to ensure accurate, constraint-aware, and personalized recommendations.

---

## 🚀 Key Features

* 🔍 **Context-Aware Retrieval (RAG)**

  * Retrieves relevant courses and syllabus content using vector embeddings.

* 🧠 **Intelligent Query Understanding**

  * Extracts:

    * Career goals (e.g., Data Scientist)
    * Completed courses
    * Constraints (credits, preferences)

* ⚙️ **Rule-Based Reasoning Engine**

  * Validates prerequisites
  * Ensures eligibility
  * Handles prerequisite chains
  * Enforces credit limits

* 🎯 **Goal-Oriented Recommendations**

  * Aligns course selection with career skill requirements

* 📊 **Structured JSON Output**

  * Returns recommendations in a strict, machine-readable format

* 🚫 **No Hallucination Guarantee**

  * All outputs are grounded in the provided academic dataset

---

## 🏗️ System Architecture

1. **Query Processing**

   * Extracts structured intent from user input

2. **Retrieval Layer**

   * Uses embeddings to fetch relevant courses and syllabus data

3. **Reasoning Engine**

   * Applies academic rules:

     * Prerequisite validation
     * Credit constraints
     * Eligibility filtering

4. **Ranking Module**

   * Ranks courses based on:

     * Career relevance
     * Eligibility
     * Learning progression

5. **Response Generation**

   * Produces structured JSON output with explanations

---

## 🧩 Tech Stack

* **Backend:** Python, FastAPI
* **RAG Framework:** LangChain
* **Vector Database:** FAISS
* **LLM:** OpenAI / LLM APIs
* **Data:** JSON (course catalog), PDFs (syllabus)

---

## 📥 Input Example

> "I want to become a data scientist. I have completed CS101 and MATH101. What courses should I take next semester?"

---

## 📤 Output Example

```json
{
  "recommended_courses": [
    {
      "course_id": "CS201",
      "course_name": "Machine Learning",
      "reason": "Aligned with data science goal and prerequisites satisfied",
      "prerequisite_status": "eligible"
    }
  ],
  "excluded_courses": [
    {
      "course_id": "CS301",
      "reason": "Missing prerequisite: CS201"
    }
  ],
  "credit_summary": {
    "total_credits": 4,
    "max_allowed": 12
  }
}
```

---

## ⚠️ Constraints Handled

* Prerequisite dependencies
* Maximum credit limits
* User preferences (e.g., avoid math-heavy courses)
* Career alignment

---

## 🔐 Security Considerations

* Prevents hallucinated recommendations
* Restricts responses to known academic data
* Input validation and controlled LLM usage
* API keys secured via environment variables

---

## 🌟 Future Enhancements

* Personalized recommendations using student performance data
* Integration with live academic platforms (timetable, quizzes)
* UI dashboard for course planning
* Multi-user role-based access (student, faculty, admin)

---

## 💡 Motivation

Students often struggle to navigate academic pathways due to fragmented information and complex prerequisite structures. This project aims to bridge that gap by providing an intelligent, explainable, and reliable academic advising system.

---

