"""Shared text preprocessing for the notebook AND the Streamlit app.

Keeping ONE copy of this code guarantees that training and prediction
clean emails in exactly the same way.
"""
import re
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Keep negations: "not", "no", "never" carry meaning in emails.
NEGATIONS = {"no", "nor", "not", "never"}
STOP_WORDS = frozenset(ENGLISH_STOP_WORDS) - NEGATIONS

# Words that only identify the *source* of the dataset (Enron Corp., Houston gas
# trading) and say nothing about spam. If the model sees them it can "cheat":
# Enron words -> ham, everything else -> spam. We mask them before training.
SOURCE_WORDS = {
    "enron", "ect", "hou", "hpl", "hplc", "hplno", "hplnol", "sitara", "daren",
    "nom", "noms", "nomination", "nominations", "mmbtu", "mmbtud", "ena", "lng",
    "corp", "cc", "pm", "am", "forwarded", "original", "message", "houston",
    "texas", "tx", "meter", "meters", "xls", "vince", "kaminski", "ken", "jerry",
    "clickathome", "neon", "txu", "ees", "eott", "pec", "wellhead", "gas",
    "volumes", "pipeline",
}

_MONTHS = ("january|february|march|april|may|june|july|august|september|"
           "october|november|december|jan|feb|mar|apr|jun|jul|aug|sep|sept|"
           "oct|nov|dec")
_DAYS = "monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
_DATE_WORDS = re.compile(rf"\b({_MONTHS}|{_DAYS})\b")
_EMAIL_ADDR = re.compile(r"\S+@\S+")
_URL = re.compile(r"http\S+|www\.\S+")


def clean_text(text: str) -> str:
    """Turn one raw email into a string of useful lowercase words."""
    text = str(text).lower()
    text = re.sub(r"^\s*subject\s*:", " ", text)        # constant 'subject:' tag
    text = _URL.sub(" urltoken ", text)                  # links -> one token
    text = _EMAIL_ADDR.sub(" emailtoken ", text)         # addresses -> one token
    text = re.sub(r"\s*'\s*t\b", " not", text)           # don ' t -> don not
    text = text.replace("$", " dollar ")
    text = _DATE_WORDS.sub(" ", text)                    # months / weekdays are Enron-era noise
    text = re.sub(r"\b(19|20)\d\d\b", " ", text)         # years (2000, 2001 ... = dataset era)
    text = re.sub(r"\d+", " number ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = [w for w in text.split()
              if w not in STOP_WORDS and w not in SOURCE_WORDS and len(w) > 1]
    return " ".join(tokens)
