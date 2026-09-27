import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# advisor.py is imported as a flat module (mirrors how `streamlit run app/app.py`
# and `python app/advisor.py` add the script's own directory to sys.path).
APP_DIR = Path(__file__).resolve().parent.parent / "app"
sys.path.insert(0, str(APP_DIR))

os.environ.setdefault("GROQ_API_KEY", "test-key-not-used")


@pytest.fixture(scope="session")
def advisor():
    """
    Import advisor.py with its network/model-loading dependencies mocked out.

    advisor.py talks to Groq and loads a local embedding model + FAISS index
    at import time. None of that is needed to test the deterministic
    filtering/matching logic, so it's mocked here to keep these tests fast,
    offline, and independent of a real GROQ_API_KEY.
    """
    with patch("langchain_huggingface.HuggingFaceEmbeddings") as mock_embeddings, \
         patch("langchain_community.vectorstores.FAISS.load_local") as mock_faiss_load, \
         patch("langchain_groq.ChatGroq") as mock_chat_groq:

        mock_embeddings.return_value = MagicMock()
        mock_faiss_load.return_value = MagicMock()
        # Default "no match" response so any LLM-fallback path in
        # enrich_completed_list resolves deterministically instead of
        # returning an un-JSON-parseable MagicMock.
        mock_chat_groq.return_value.invoke.return_value.content = "[]"

        sys.modules.pop("advisor", None)
        import advisor as advisor_module
        yield advisor_module
