import json
from typing import List, Dict
from collections import Counter

import nltk
import spacy
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords


class TextPreprocessor:
    """
    Train once on training data, then reuse everywhere.
    """

    def __init__(self, min_freq: int = 1):
        self.min_freq = min_freq
        self.vocab: Dict[str, int] | None = None
        self.nlp = None
        self.stop_words = None

    # Setup ---------------------------------------------------
    def __len__(self):
        return len(self.vocab) if self.vocab else 0
    def _setup(self):
        nltk.download("punkt")
        nltk.download('punkt_tab')
        nltk.download("stopwords")

        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

        self.stop_words = set(stopwords.words("english"))

    # Core steps ----------------------------------------------

    def _tokenize(self, text: str) -> List[str]:
        return [
            w for w in word_tokenize(text.lower())
            if w.isalpha() and w not in self.stop_words
        ]

    def _lemmatize(self, tokens: List[str]) -> List[str]:
        doc = self.nlp(" ".join(tokens))
        return [t.lemma_ for t in doc]

    # ------------------
    # FIT (TRAIN ONLY)
    # ------------------

    def fit(self, df: pd.DataFrame, text_column: str = "text"):
        self._setup()

        all_tokens = []

        for text in df[text_column]:
            tokens = self._tokenize(text)
            lemmas = self._lemmatize(tokens)
            all_tokens.extend(lemmas)

        counter = Counter(all_tokens)

        self.vocab = {"<PAD>": 0, "<UNK>": 1}
        for word, freq in counter.items():
            if freq >= self.min_freq:
                self.vocab[word] = len(self.vocab)

        return self

    # ------------------
    # TRANSFORM (EVERYWHERE)
    # ------------------

    def transform(self, df: pd.DataFrame, text_column: str = "text") -> pd.DataFrame:
        if self.vocab is None:
            raise RuntimeError("Preprocessor must be fitted before calling transform().")

        df = df.copy()

        df["tokens"] = df[text_column].apply(self._tokenize)
        df["lemmas"] = df["tokens"].apply(self._lemmatize)
        df["numericalized"] = df["lemmas"].apply(
            lambda x: [self.vocab.get(t, self.vocab["<UNK>"]) for t in x]
        )

        return df

    # ------------------
    # SERIALIZATION
    # ------------------

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.vocab, f)

    def load(self, path: str):
        with open(path) as f:
            self.vocab = json.load(f)
        self._setup()
        return self
