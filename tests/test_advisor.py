# =============================================================================
# Tests for the deterministic filtering/matching logic in app/advisor.py.
#
# This is the logic that keeps the system from hallucinating ineligible
# courses (see README "Design Decisions & Tradeoffs"): prerequisite checks,
# career relevance, and course-name/code enrichment all run in Python before
# the LLM ever sees the data. These tests exist to back that claim.
# =============================================================================


# ── is_course_satisfied ──────────────────────────────────────────────────────

def test_is_course_satisfied_direct_substring_match(advisor):
    completed = ["BCS304"]
    prereq = "Data Structure and Applications (BCS304)"
    assert advisor.is_course_satisfied(completed, prereq) is True


def test_is_course_satisfied_returns_false_when_missing(advisor):
    completed = ["BCS999"]
    prereq = "Data Structure and Applications (BCS304)"
    assert advisor.is_course_satisfied(completed, prereq) is False


def test_is_course_satisfied_reverse_substring_match(advisor):
    # The completed list holds a longer identity string; the prereq is just
    # the bare code, so the match has to run in the reverse direction.
    completed = ["DATA STRUCTURE AND APPLICATIONS (BCS304)"]
    prereq = "BCS304"
    assert advisor.is_course_satisfied(completed, prereq) is True


def test_is_course_satisfied_extracts_code_from_brackets(advisor):
    completed = ["BPLCK105B"]
    prereq = "Introduction to Python Programming (BPLCK105B/205B)"
    assert advisor.is_course_satisfied(completed, prereq) is True


# ── is_career_relevant ───────────────────────────────────────────────────────

def test_is_career_relevant_true_when_no_keywords(advisor):
    # Unknown career goal -> career_keywords is [] -> everything passes
    # through and the LLM applies its own judgment downstream.
    course = {"course_id": "X1", "course_name": "Anything", "outcomes": "anything"}
    assert advisor.is_career_relevant(course, []) is True


def test_is_career_relevant_single_word_keyword_needs_two_occurrences(advisor):
    course_one_mention = {
        "course_id": "X1",
        "course_name": "Intro to Programming",
        "outcomes": "Covers basic python syntax.",
    }
    course_two_mentions = {
        "course_id": "X2",
        "course_name": "Python Programming",
        "outcomes": "Deep dive into python. Builds real python projects.",
    }
    # One passing mention shouldn't be enough to flag the course as "about"
    # that topic — this is what stops a course that merely name-drops a
    # skill from being treated as career-relevant.
    assert advisor.is_career_relevant(course_one_mention, ["python"]) is False
    assert advisor.is_career_relevant(course_two_mentions, ["python"]) is True


def test_is_career_relevant_multi_word_phrase_match(advisor):
    course = {
        "course_id": "X3",
        "course_name": "Neural Networks",
        "outcomes": "Introduces machine learning fundamentals and training.",
    }
    assert advisor.is_career_relevant(course, ["machine learning"]) is True


def test_is_career_relevant_false_when_nothing_matches(advisor):
    course = {
        "course_id": "X4",
        "course_name": "History of Art",
        "outcomes": "Studies painting techniques.",
    }
    assert advisor.is_career_relevant(course, ["machine learning", "python"]) is False


# ── enrich_completed_list ────────────────────────────────────────────────────
# Uses the real Data/courses.json shipped with the repo (e.g. BCS304 "Data
# Structure and Applications", BPHYS102/202 "Applied Physics for CSE Stream").

def test_enrich_completed_list_exact_code_match(advisor):
    enriched = advisor.enrich_completed_list(["BCS304"])
    assert "BCS304" in enriched


def test_enrich_completed_list_exact_name_match(advisor):
    enriched = advisor.enrich_completed_list(["DATA STRUCTURE AND APPLICATIONS"])
    assert "BCS304" in enriched


def test_enrich_completed_list_substring_match(advisor):
    enriched = advisor.enrich_completed_list(["STRUCTURE AND APPLICATIONS"])
    assert "BCS304" in enriched


def test_enrich_completed_list_partial_code_match(advisor):
    # "BPHYS102" is contained within the slash-joined course code "BPHYS102/202"
    enriched = advisor.enrich_completed_list(["BPHYS102"])
    assert "BPHYS102/202" in enriched


def test_enrich_completed_list_keeps_unresolvable_entries_as_typed(advisor):
    # Nothing in courses.json matches this, and the mocked LLM fallback
    # (conftest.py) returns "[]" — the function should degrade gracefully
    # and keep the original entry rather than dropping or crashing on it.
    result = advisor.enrich_completed_list(["Some Totally Unknown Course XYZ"])
    assert "Some Totally Unknown Course XYZ" in result
