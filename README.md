# 🎓 AI Academic Advisor — Complete Project Documentation

> A RAG-based intelligent system that recommends courses to students based on their career goals, completed courses, and credit constraints — with zero hallucinations.

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Why This Project Exists](#-why-this-project-exists)
3. [Tech Stack & Why Each Tool Was Chosen](#-tech-stack--why-each-tool-was-chosen)
4. [System Architecture — The Big Picture](#-system-architecture--the-big-picture)
5. [File Structure](#-file-structure)
6. [Complete Pipeline Walkthrough (Stage by Stage)](#-complete-pipeline-walkthrough-stage-by-stage)
7. [Every Function Explained](#-every-function-explained)
8. [The Data Layer](#-the-data-layer)
9. [The UI Layer (app.py)](#-the-ui-layer-apppy)
10. [Setup & Installation](#-setup--installation)
11. [Example Input & Output](#-example-input--output)
12. [Design Decisions & Tradeoffs](#-design-decisions--tradeoffs)
13. [Future Enhancements](#-future-enhancements)

---

## 🧠 Project Overview

The AI Academic Advisor is a **Retrieval-Augmented Generation (RAG)** system that acts like a personal academic counselor. A student tells it:

- What career they want (e.g., "AI Engineer")
- What courses they've already completed
- How many credits they can take this semester

And the system replies with:

- A list of **courses they should enroll in right now**
- A **roadmap** showing which courses unlock which future courses
- An explanation of **why each course matters** for their specific career

The key design goal was: **never recommend a course the student isn't eligible for, and never make things up.** Every recommendation is grounded in real data.

---

## 💡 Why This Project Exists

Students at universities often face a common problem:

- The course catalog is huge and hard to navigate
- Prerequisite chains are complex ("I need A before B before C before the course I actually want")
- Academic advisors are not always available
- Generic chatbots hallucinate courses that don't exist

This project solves all of that by combining the **reasoning power of an LLM** with **deterministic rule-based filtering** to give advice that is both intelligent and reliable.

---

## 🛠 Tech Stack & Why Each Tool Was Chosen

| Tool | Purpose | Why This and Not Something Else |
|---|---|---|
| **Python** | Core language | Universal for ML/AI work; rich ecosystem |
| **LangChain** | RAG orchestration framework | Makes it easy to connect embeddings, vector stores, and LLMs in a pipeline |
| **FAISS** | Vector database for semantic search | Fast, local, no cloud needed; perfect for a fixed course catalog |
| **HuggingFace `all-MiniLM-L6-v2`** | Embedding model | Small (80MB), fast, and surprisingly accurate for semantic matching; free to use |
| **Groq + LLaMA 3.3 70B** | The LLM for reasoning and roadmap generation | Groq provides extremely fast inference; LLaMA 3.3 70B is powerful enough to generate coherent academic advice |
| **Streamlit** | Web UI | Lets you build a clean web app in pure Python — no HTML/CSS/JS needed |
| **python-dotenv** | API key management | Keeps secrets out of source code |
| **JSON files** | Data storage for courses and careers | Simple, human-readable, easy to edit without a database |

---

## 🏗 System Architecture — The Big Picture

Here is the full flow from a student's question to a structured recommendation:

```
Student Input (career goal + completed courses + credit limit + question)
        │
        ▼
┌─────────────────────────────────────────┐
│  STAGE A: Career Keyword Lookup         │
│  career.json → list of skill keywords   │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  STAGE B: Course History Enrichment     │
│  "Intro to Python" → "BPLCK105B/205B"   │
│  Python exact match first, LLM fallback │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  STAGE C: FAISS Semantic Retrieval      │
│  Query → embeddings → top-k similar     │
│  courses from vector database           │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  STAGE D: Python-Side Filtering         │
│  ✅ Are prerequisites met?              │
│  ✅ Is the course career-relevant?      │
│  ✅ Will it fit within credit limit?    │
│  ✅ Not already completed?              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  STAGE E: LLM Roadmap Generation        │
│  Filtered pool → LLaMA 3.3 70B          │
│  → Structured JSON roadmap              │
└─────────────────┬───────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────┐
│  STAGE F: Display                       │
│  JSON → Pretty terminal output OR       │
│  Streamlit UI with styled cards         │
└─────────────────────────────────────────┘
```

**The crucial insight:** The LLM only sees *already-filtered* data. All the hard logic (prerequisite checking, credit limits, career matching) is done by Python code — not by the AI. The AI's job is only to rank, explain, and write the narrative. This is why the system never hallucinates ineligible courses.

---

## 📁 File Structure

```
AI-Academic-Advisor/
│
├── app/
│   ├── app.py          ← Streamlit web UI
│   ├── advisor.py      ← Core pipeline (all 6 stages)
│   └── ingest.py       ← One-time script to build vector database
│
├── Data/
│   ├── courses.json    ← Full course catalog with prerequisites, credits, outcomes
│   └── career.json     ← Maps career titles to required skill keywords
│
├── faiss_index/        ← Auto-generated by ingest.py (do not edit manually)
│   ├── index.faiss
│   └── index.pkl
│
├── .env                ← Your API keys (never commit to git)
└── req.txt             ← Python dependencies
```

---

## 🔬 Complete Pipeline Walkthrough (Stage by Stage)

### Stage A — Career Keyword Lookup

**What it does:** Converts a free-text career goal like "AI Engineer" into a list of skill keywords like `["machine learning", "deep learning", "python", "statistics", "algorithms"]`.

**Why this matters:** The system needs to know *what skills* a career requires so it can filter out irrelevant courses. Without this, a student aiming for AI might get recommended web development courses.

**How it works:** It reads `career.json`, splits both the student's goal and each career key into words, and counts how many words overlap. The career with the most overlap wins. This approach requires zero external libraries — no fuzzy matching needed.

**Example:**
- Student types: `"AI Engineer"`
- Words: `{"ai", "engineer"}`
- `"ai_engineer"` key → words: `{"ai", "engineer"}` → overlap = 2 → **match!**
- Returns: `["machine learning", "deep learning", "python", "statistics", "algorithms"]`

---

### Stage B — Completed Course Enrichment

**What it does:** Takes what the student typed (e.g., `"Intro to Python"`, `"BCS301"`) and expands it with official course codes so prerequisite checks always work.

**Why this matters:** The prerequisites in `courses.json` look like:
`"Introduction to Python Programming (BPLCK105B/205B)"`

If a student typed "Intro to Python", a direct equality check would fail. By adding the course code `BPLCK105B/205B` to the student's completed list, all future checks become simple and reliable.

**Two-phase approach (Python first, LLM only as fallback):**

| Phase | Method | Speed | Cost |
|---|---|---|---|
| Phase 1 | Python substring/exact matching | Instant | Free |
| Phase 2 | LLM semantic matching | ~2 sec | API call |

The LLM is only called for entries Python couldn't resolve (abbreviations, typos). This keeps the system fast and cheap.

---

### Stage C — FAISS Semantic Retrieval

**What it does:** Finds courses that are semantically relevant to the student's query and career goal by searching the vector database.

**Why FAISS:** Each course was converted into a vector (a list of 384 numbers) when `ingest.py` was run. FAISS stores these vectors and can find the most similar ones to any new query vector in milliseconds — even with hundreds of courses.

**Two retrieval strategies run in parallel:**

1. **Primary search** — Takes the student's question + career goal, builds an enriched query, and finds the top 30 semantically similar courses.

2. **Foundation sweep** — Always retrieves fundamental courses (math, programming basics) using 3 different angle queries. This prevents a vague question like "what next?" from missing foundational courses that every career path needs.

Both results are merged and deduplicated by course ID.

---

### Stage D — Python-Side Filtering

**What it does:** Runs every retrieved course through 6 sequential checks. A course must pass ALL checks to enter the recommendation pool.

**The 6 checks (in order):**

| # | Check | Why This Order |
|---|---|---|
| 1 | Already completed? | Skip immediately — cheapest check |
| 2 | Prerequisites met? | Moves blocked courses to "excluded" list for roadmap use |
| 3 | Is it a 0-credit course? | Drop labs/activities from recommendations |
| 4 | Career relevant? | Drop courses unrelated to the student's goal |
| 5 | Within credit limit? | Stop adding courses when semester limit is reached |
| 6 | Duplicate name? | Remove same-course with different codes (e.g., BCS405A/B) |

**The excluded list is kept on purpose** — it feeds the roadmap in Stage E, showing students exactly which courses they need to complete first.

---

### Stage E — LLM Roadmap Generation

**What it does:** Takes the filtered pool of eligible and blocked courses and asks LLaMA 3.3 70B to build a structured JSON roadmap.

**What the LLM is responsible for (and nothing else):**
- Ranking the eligible courses by career impact
- Writing a one-line explanation for each recommendation
- Building a 3-hop roadmap chain showing the path to the career goal
- Writing an encouraging message

**What the LLM is NOT allowed to do:**
- Add courses not in the provided pool
- Change eligibility decisions
- Deviate from the strict JSON schema

Temperature is set to 0 so outputs are deterministic and repeatable.

---

### Stage F — Display

**What it does:** Parses the JSON from Stage E and renders it cleanly — either as terminal output (advisor.py) or as styled cards in Streamlit (app.py).

---

## 🔧 Every Function Explained

### `ingest.py`

#### `create_vector_db()`
**What:** Reads every course from `courses.json`, converts each one into a text document, generates embeddings, and saves the FAISS index to disk.

**Why:** This only needs to run once (or when course data changes). By saving the index locally, the main advisor doesn't have to rebuild it on every startup — that would take 30+ seconds every time.

**Key design choice:** The `page_content` string deliberately includes the course code (`"Course Code: BCS301."`) so that when a student asks specifically about a course code, FAISS can find it. Without this, `BCS301` might not appear in results for a query like "BCS301 courses."

---

### `advisor.py`

#### `get_career_keywords(career_goal: str) → list[str]`
**What:** Looks up a free-text career goal in `career.json` and returns a list of required skill keywords.

**Why:** Converts a vague goal into measurable, searchable attributes. The word-overlap scoring means it handles variations like "AI Engineer", "Artificial Intelligence Engineer", and "engineer for AI" all correctly.

**Returns `[]` on no match** — this is intentional. When the career is unknown, all courses pass through and the LLM uses its own judgment.

---

#### `enrich_completed_list(completed_names: list[str]) → list[str]`
**What:** Expands informal course names into official course codes.

**Why:** Prerequisite checking requires exact or substring matching. Enrichment is the bridge between how humans write course names and how the system stores them.

**5 Python matching strategies (tried in order):**
1. Exact course code match (`"BCS301"` → `"BCS301"`)
2. Exact course name match (full copy-paste)
3. Substring of course name (student's input is inside the official name)
4. Partial code match (`"BPLCK105B"` inside `"BPLCK105B/205B"`)
5. Split code match (first part of a slash-separated code)

**LLM fallback:** Only triggered for entries that fail all 5 Python strategies. Uses batch processing to minimize API calls.

---

#### `build_retrieval_query(user_query, career_goal, career_keywords) → str`
**What:** Enriches the raw user query with career context before sending it to FAISS.

**Why:** A query like "what to study next?" has no career signal. Adding keywords like "machine learning, deep learning, python" to it steers the FAISS embedding vector toward relevant courses.

**Example transformation:**
- Input: `"what should I study next?"`
- Output: `"AI Engineer courses: machine learning, deep learning. what should I study next?"`

---

#### `get_foundation_sweep_docs() → list`
**What:** Always retrieves foundational courses regardless of the user's query.

**Why:** Critical courses like `BCS301 (Math for CS)` have no prerequisites and are essential for AI careers, but a vague query might never retrieve them. Three different sweep queries from different angles guarantee they always enter the pipeline.

**3 sweep angles:**
1. AI/ML vocabulary
2. Mathematics and statistics
3. Programming and data fundamentals

---

#### `retrieve_candidate_docs(user_query, career_goal, career_keywords) → list`
**What:** Merges the primary semantic search and the foundation sweep into one deduplicated list.

**Why:** Deduplication by `course_id` ensures the same course doesn't appear twice with slightly different similarity scores, which would confuse the filtering stage.

---

#### `is_course_satisfied(completed_upper, prereq_string) → bool`
**What:** Checks whether a single prerequisite string is satisfied by the student's completed course list.

**3 strategies:**
1. Direct substring — `"PYTHON"` inside `"INTRO TO PYTHON PROGRAMMING"`
2. Code extraction — pulls codes like `"BPLCK105B"` out of brackets and compares
3. Reverse substring — checks if the prerequisite string contains a completed item

**Why deterministic:** Prerequisite checks must be 100% reliable. Fuzzy/probabilistic matching here would be dangerous (could approve a student for a course they're not ready for). The enrichment stage (Stage B) does the hard semantic work so this function stays simple.

---

#### `is_career_relevant(course, career_keywords) → bool`
**What:** Checks whether a course is relevant to the student's target career.

**How it works:** Builds a searchable string from course name + outcomes, then tests each career keyword against it.

**Important nuance — single-word vs multi-word keywords:**
- Single-word keyword (e.g., `"python"`): Must appear **at least twice** as a whole word. This prevents a course that merely *mentions* Python in passing from being flagged as a Python course.
- Multi-word keyword (e.g., `"machine learning"`): Checks the full phrase first, then checks significant individual words (length > 5).

**Returns `True` if ANY keyword matches** — OR logic, not AND. This is intentional because a course only needs to address one skill area to be worth considering.

---

#### `deduplicate_pool(eligible_pool) → list`
**What:** Removes courses with identical names from the eligible pool.

**Why:** Courses like `BCS405A` and `BCS405B` are both called "Discrete Mathematical Structures." Without deduplication, both would appear in recommendations, confusing the student. Keeps the first occurrence.

---

#### `filter_candidates(docs, completed_upper, career_keywords, credit_limit) → tuple`
**What:** The master filtering function. Runs every retrieved document through all 6 checks and returns two lists: eligible courses and excluded courses.

**Key optimization — sort before filtering:** Documents are sorted by number of prerequisites (ascending) before filtering. This means zero-prerequisite foundational courses are processed first and are more likely to make it into the pool before the credit limit is hit.

---

#### `build_llm_response(eligible_pool, excluded, career_goal, career_keywords) → str`
**What:** Constructs the system prompt and user prompt, sends them to LLaMA 3.3 70B via Groq, and returns the raw JSON string.

**System prompt enforces:**
- Strict JSON schema (no improvised keys)
- Maximum 3 courses in `enroll_now`
- Minimum 2 hops in the roadmap chain
- No 0-credit or off-topic courses
- Graceful handling when the eligible pool is empty

**Why temperature=0:** Academic advice should be consistent. The same student profile should always get the same recommendation. Random variation in advice would be confusing and unprofessional.

---

#### `display_response(raw_response) → None`
**What:** Parses the JSON and prints it to the terminal in a clean, readable format.

**Why it strips markdown fences:** LLMs sometimes wrap JSON in ` ```json ``` ` blocks even when instructed not to. The `re.sub` call handles this gracefully so parsing never fails due to formatting.

---

#### `academic_advisor(query, completed_courses, career_goal, credit_limit) → None`
**What:** The main entry point that ties all 6 stages together for a single query.

**Session loop:** The CLI version maintains a session for up to 10 minutes of inactivity and supports mid-session additions of forgotten courses via the "forgot" command.

---

### `app.py` (Streamlit UI)

#### `load_advisor_resources()`
**What:** Imports the `advisor` module and caches it with `@st.cache_resource`.

**Why caching:** Without this, every time Streamlit rerenders the page (which happens on every user interaction), it would reload the FAISS index and embedding model — taking 30 seconds each time. Caching means this happens exactly once per server session.

#### `render_response(raw: str)`
**What:** Parses the LLM's JSON and renders it with styled HTML cards in Streamlit.

**Three card types:**
- `result-card` (blue) — recommended courses
- `roadmap-card` (green) — roadmap steps  
- `warn-card` (yellow) — warnings when no courses are available

**Why `unsafe_allow_html=True`:** Streamlit's native components don't support the level of visual customization needed for polished cards. Custom HTML/CSS gives control over colors, borders, badges, and typography.

---

## 📊 The Data Layer

### `courses.json`
Each course entry contains:

| Field | Purpose |
|---|---|
| `course_code` | Unique ID (e.g., `"BCS602"`) |
| `name` | Human-readable name |
| `description` | What the course is about |
| `prerequisites` | List of prerequisite strings (can be empty) |
| `credits` | Credit weight |
| `tags` | Keywords for manual categorization |
| `topics` | Module-level topic list |
| `course_outcomes` | What students will be able to do after the course |

The `course_outcomes` field is especially important — it's included in the FAISS embedding, which means semantic searches can match based on *what a course teaches* not just its name.

### `career.json`
Maps career titles to required skill keywords:
```json
"ai_engineer": {
  "required_skills": ["machine learning", "deep learning", "python", "statistics", "algorithms"]
}
```

**Adding a new career requires zero code changes** — just add a new entry to this file. This is a key scalability design choice.

---

## 🖥 The UI Layer (app.py)

The Streamlit app has two main sections:

**Sidebar (persistent profile):**
- Career goal text input
- Completed courses text area
- Credit limit slider
- "Forgot a course?" mid-session adder
- Reset session button

**Main area:**
- Query input form
- Chat history (most recent first, each in a collapsible expander)
- Styled output cards for recommendations and roadmap

**Session state management:**
- `chat_history` — list of (query, response) tuples persisted across rerenders
- `enriched_completed` — enriched course list computed once and cached to avoid repeated LLM calls

---

## ⚙️ Setup & Installation

### 1. Clone the Repository
```bash
git clone https://github.com/BiswasNehaa/AI-Academic-Advisor.git
cd AI-Academic-Advisor
```

### 2. Create a Virtual Environment
```bash
python -m venv venv

# Windows:
.\venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r req.txt
```

### 4. Set Up API Keys
Create a `.env` file in the root directory:
```
GROQ_API_KEY=your_groq_api_key_here
```
Get a free key at [console.groq.com](https://console.groq.com)

### 5. Build the Vector Database (Run Once)
```bash
python app/ingest.py
```
This reads `courses.json`, generates embeddings, and saves a `faiss_index/` folder. Only needs to be re-run when course data changes.

### 6. Launch the App

**Web UI:**
```bash
streamlit run app/app.py
```

**Terminal (CLI):**
```bash
python app/advisor.py
```

---

## 📥 Example Input & Output

**Student Profile:**
- Career Goal: `AI Engineer`
- Completed: `Introduction to Python Programming, Mathematics for Computer Science`
- Credit Limit: `12`
- Query: `"What should I study next semester?"`

**Output (JSON from LLM):**
```json
{
  "message": "You have built a strong foundation — you're ready to dive into data and algorithms!",
  "enroll_now": [
    {
      "course_id": "BCS358A",
      "course_name": "Data Analytics with Python",
      "credits": 3,
      "why": "Directly applies Python skills to data analysis using Pandas, NumPy, and Scikit-Learn — core tools for any AI Engineer."
    },
    {
      "course_id": "BCS401",
      "course_name": "Analysis and Design of Algorithms",
      "credits": 4,
      "why": "Algorithms are the backbone of ML systems; required prerequisite for Machine Learning (BCS602)."
    }
  ],
  "unlock_next": [
    {
      "complete_first": "Analysis and Design of Algorithms (BCS401)",
      "this_will_unlock": "Principles of Artificial Intelligence (BCS515B)",
      "which_then_unlocks": "Machine Learning (BCS602)"
    },
    {
      "complete_first": "Machine Learning (BCS602)",
      "this_will_unlock": "Deep Learning (BCS714A)",
      "which_then_unlocks": "Natural Language Processing (BCS714B)"
    }
  ]
}
```

---

## 🤔 Design Decisions & Tradeoffs

| Decision | Alternative Considered | Why This Choice Won |
|---|---|---|
| Python handles filtering, LLM only ranks | Let LLM do everything | LLMs are unreliable for hard constraint checking (credit limits, exact prerequisite matching). Python is deterministic. |
| FAISS local vector store | Pinecone, ChromaDB | No cloud dependency, no cost, fast enough for a fixed course catalog |
| Two-phase enrichment (Python + LLM) | LLM only | Reduces API calls by ~80%, faster response, lower cost |
| Sort docs by prereq count before filtering | Random order | Foundational courses (0 prereqs) get processed first and are less likely to be cut by the credit cap |
| `temperature=0` for LLM | Higher temperature | Academic advice must be consistent and reproducible |
| Foundation sweep in retrieval | Query-only retrieval | Prevents vague queries from missing critical foundational courses |

---

## 🌟 Future Enhancements

- **Student performance data integration** — weight recommendations by historical grades in related subjects
- **Multi-semester planning** — generate a full 2-year roadmap, not just one semester
- **Live timetable integration** — filter by available class times
- **Faculty ratings** — incorporate professor ratings into the ranking
- **Multi-user roles** — separate views for students, faculty, and academic coordinators
- **Feedback loop** — let students rate recommendations to improve future suggestions
- **Expanded career catalog** — add more career profiles to `career.json`

---

## 🔐 Security Notes

- API keys are stored in `.env` and never hardcoded
- The FAISS index is built locally — no student data sent to external services during retrieval
- LLM calls include only course data and filtered pools — no personally identifiable information
- `allow_dangerous_deserialization=True` is safe here because the index is built by us, not loaded from an untrusted source

---

*Built with LangChain · FAISS · HuggingFace Embeddings · Groq (LLaMA 3.3 70B) · Streamlit*
