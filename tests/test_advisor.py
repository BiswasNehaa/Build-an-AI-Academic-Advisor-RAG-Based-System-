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


def test_is_career_relevant_single_shared_word_is_not_enough_for_a_phrase(advisor):
    # Regression test: a computer organization course whose outcomes mention
    # "machine instructions" was incorrectly flagged as "machine learning"
    # relevant because the old logic matched on ANY significant word in the
    # keyword phrase rather than ALL of them.
    course = {
        "course_id": "BCS302",
        "course_name": "Digital Design and Computer Organization",
        "outcomes": "Describe the fundamentals of machine instructions, addressing modes and processor performance.",
    }
    assert advisor.is_career_relevant(course, ["machine learning"]) is False


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


# ── validate_course_codes ────────────────────────────────────────────────────
# Regression coverage for a real bug: the LLM's roadmap "which_then_unlocks"
# hop has no grounded data behind it and would invent plausible-looking but
# fake course codes (e.g. "BCS500"). This is the Python-side safety net that
# catches that instead of trusting the LLM's output at face value.

def test_validate_course_codes_drops_fabricated_enroll_now_entry(advisor):
    data = {
        "enroll_now": [
            {"course_id": "BCS304", "course_name": "Data Structure and Applications"},
            {"course_id": "BCS999", "course_name": "Totally Made Up Course"},
        ],
        "unlock_next": [],
    }
    result = advisor.validate_course_codes(data)
    ids = [c["course_id"] for c in result["enroll_now"]]
    assert ids == ["BCS304"]


def test_validate_course_codes_blanks_fabricated_third_hop_only(advisor):
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "Data Structure and Applications (BCS304)",
                "this_will_unlock": "Analysis and Design of Algorithms (BCS401)",
                "which_then_unlocks": "Advanced Algorithms (BCS500)",  # fabricated
            }
        ],
    }
    result = advisor.validate_course_codes(data)
    assert len(result["unlock_next"]) == 1
    step = result["unlock_next"][0]
    assert step["complete_first"] == "Data Structure and Applications (BCS304)"
    assert step["this_will_unlock"] == "Analysis and Design of Algorithms (BCS401)"
    assert step["which_then_unlocks"] == ""


def test_validate_course_codes_drops_step_with_fabricated_first_hop(advisor):
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "Totally Fake Prereq (BCS200)",
                "this_will_unlock": "Analysis and Design of Algorithms (BCS401)",
                "which_then_unlocks": "",
            }
        ],
    }
    result = advisor.validate_course_codes(data)
    assert result["unlock_next"] == []


def test_validate_course_codes_keeps_steps_without_any_code_claim(advisor):
    # Free-text descriptions with no course code aren't verifiable against
    # the catalog, so they pass through unchanged rather than being dropped.
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "Basics of C programming concepts",
                "this_will_unlock": "Data Structure and Applications (BCS304)",
                "which_then_unlocks": "",
            }
        ],
    }
    result = advisor.validate_course_codes(data)
    assert len(result["unlock_next"]) == 1


def test_validate_course_codes_drops_step_with_false_unlock_claim(advisor):
    # Regression test for a real bug observed on the deployed app: the LLM
    # claimed "Digital Design and Computer Organization" (BCS302, which has
    # no prerequisites and isn't a prerequisite of anything) would unlock
    # "Data Structures Laboratory" (BCSL305), whose real prerequisite is
    # "C Programming Concepts". Both course names/codes are real, but the
    # causal claim between them is false -- is_grounded() alone (checking
    # codes exist) wouldn't have caught this.
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "Digital Design and Computer Organization",
                "this_will_unlock": "Data Structures Laboratory (BCSL305)",
                "which_then_unlocks": "",
            }
        ],
    }
    result = advisor.validate_course_codes(data)
    assert result["unlock_next"] == []


def test_validate_course_codes_keeps_step_with_true_unlock_claim(advisor):
    # BCS401's real prerequisites include "Data Structure and Applications (BCS304)".
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "Data Structure and Applications (BCS304)",
                "this_will_unlock": "Analysis and Design of Algorithms (BCS401)",
                "which_then_unlocks": "",
            }
        ],
    }
    result = advisor.validate_course_codes(data)
    assert len(result["unlock_next"]) == 1


def test_validate_course_codes_normalizes_bare_code_to_name_and_code(advisor):
    # Regression test: the LLM sometimes returns a bare code for
    # this_will_unlock/which_then_unlocks (e.g. just "BCSL305") even though
    # the prompt asks for "name (code)" -- a bare code isn't meaningful to a
    # student who doesn't have the catalog memorized, so Python normalizes it.
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "C Programming Concepts",
                "this_will_unlock": "BCSL305",
                "which_then_unlocks": "",
            }
        ],
    }
    result = advisor.validate_course_codes(data)
    assert len(result["unlock_next"]) == 1
    assert result["unlock_next"][0]["this_will_unlock"] == "Data Structures Laboratory (BCSL305)"


# ── validate_course_codes fallback ───────────────────────────────────────────
# Regression test for a real bug: once the validator started actually
# enforcing real prerequisite relationships, it could end up rejecting every
# step the LLM proposed, leaving the student with an empty roadmap and no
# guidance at all. The fallback rebuilds the roadmap directly from real
# blocked-course data instead of showing nothing.

def test_validate_course_codes_falls_back_when_everything_is_rejected(advisor):
    data = {"enroll_now": [], "unlock_next": []}   # nothing survived validation
    excluded = [
        {
            "course_id": "BCSL305",
            "course_name": "Data Structures Laboratory",
            "reason": "Missing: C Programming Concepts",
        },
        {
            "course_id": "BCS304",
            "course_name": "Data Structure and Applications",
            "reason": "Missing: Basics of C programming concepts",
        },
    ]
    result = advisor.validate_course_codes(data, excluded)
    assert len(result["unlock_next"]) == 2
    step = result["unlock_next"][0]
    assert step["complete_first"] == "C Programming Concepts"
    assert step["this_will_unlock"] == "Data Structures Laboratory (BCSL305)"


def test_validate_course_codes_no_fallback_when_excluded_is_empty(advisor):
    data = {"enroll_now": [], "unlock_next": []}
    result = advisor.validate_course_codes(data, excluded=[])
    assert result["unlock_next"] == []


def test_validate_course_codes_prefers_real_llm_steps_over_fallback(advisor):
    data = {
        "enroll_now": [],
        "unlock_next": [
            {
                "complete_first": "Data Structure and Applications (BCS304)",
                "this_will_unlock": "Analysis and Design of Algorithms (BCS401)",
                "which_then_unlocks": "",
            }
        ],
    }
    excluded = [
        {
            "course_id": "BCSL305",
            "course_name": "Data Structures Laboratory",
            "reason": "Missing: C Programming Concepts",
        }
    ]
    result = advisor.validate_course_codes(data, excluded)
    # The LLM's own (valid) step should be kept, not replaced by the fallback.
    assert len(result["unlock_next"]) == 1
    assert "Analysis and Design of Algorithms" in result["unlock_next"][0]["this_will_unlock"]
