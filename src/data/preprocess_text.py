
df_train = df_train[df_train["label"] != 2]
df_test = df_test[df_test["label"] != 2]

df_train.loc[:,"label"] = df_train.loc[:,"label"].map({0: 0, 1: 1, 3: 2, 4: 3, 5: 4})
df_test.loc[:,"label"] = df_test.loc[:,"label"].map({0: 0, 1: 1, 3: 2, 4: 3, 5: 4})

# 1. Tokenize:----------------------

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download resources once
nltk.download("punkt")
nltk.download("stopwords")
nltk.download('punkt_tab')

stop_words = set(stopwords.words("english"))

# Tokenize and clean
df_train["tokens"] = df_train["text"].apply(lambda x: [
    w for w in word_tokenize(x.lower()) if w.isalpha() and w not in stop_words
])


# 2. Normalization: (Lemmatizing)-----------

import spacy

# Load English tokenizer, tagger, parser, NER and word vectors
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    print("Downloading spaCy model 'en_core_web_sm'...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")
    
def lemmatize_tokens(tokens):
    doc = nlp(" ".join(tokens))
    return [token.lemma_ for token in doc]

df_train["lemmas"] = df_train["tokens"].apply(lemmatize_tokens)

# 3. Building Vocabulary:

from collections import Counter

# Step 1: Flatten tokens
all_tokens = [tok for tokens in df_train["lemmas"] for tok in tokens]

# Step 2: Count frequencies
counter = Counter(all_tokens)

# Step 3: Build vocab dictionary
vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common())}
vocab["<PAD>"] = 0
vocab["<UNK>"] = 1

print(vocab)

# Step 4: Map sequences
df_train["numericalized"] = df_train["lemmas"].apply(lambda lemmas: [vocab.get(lem, vocab["<UNK>"]) \
                                                                     for lem in lemmas])
