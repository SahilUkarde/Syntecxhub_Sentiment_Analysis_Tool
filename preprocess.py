"""
preprocess.py
--------------
Shared text cleaning / tokenization utilities used by both the training
script and the prediction CLI, so the SAME preprocessing is applied
everywhere.
"""

import re
import string

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Compile regex patterns once for speed/readability
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
NON_ALPHA_RE = re.compile(r"[^a-z\s]")
MULTI_SPACE_RE = re.compile(r"\s+")

STOPWORDS = set(ENGLISH_STOP_WORDS)


def clean_text(text: str) -> str:
    """
    Clean a single piece of text:
      1. lowercase
      2. remove URLs
      3. remove @mentions
      4. turn #hashtags into plain words
      5. remove punctuation / digits / non-letter characters
      6. tokenize on whitespace
      7. remove stopwords and single-character tokens
      8. rejoin into a cleaned string

    Returns a cleaned, space-separated string ready for vectorization.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(" ", text)
    text = HASHTAG_RE.sub(r"\1", text)  # keep the word, drop the #
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = NON_ALPHA_RE.sub(" ", text)

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]

    return MULTI_SPACE_RE.sub(" ", " ".join(tokens)).strip()


def clean_series(series):
    """Apply clean_text to a pandas Series of raw text."""
    return series.astype(str).apply(clean_text)


if __name__ == "__main__":
    # quick manual test
    samples = [
        "I LOVE this!!! Check it out http://example.com @friend #amazing",
        "Worst product ever... 1/10, would NOT recommend.",
        "It's okay, nothing special.",
    ]
    for s in samples:
        print(f"{s!r} -> {clean_text(s)!r}")
