# corpus_prep_contentwords.py
# =========================================================
# Preprocessing pipeline for Pride & Prejudice
# Outputs: corpus_tokens.txt (space-tokenized, function words removed)
# =========================================================

import re
import urllib.request
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# --------------------- 0) NLTK resources ---------------------
nltk.download('punkt')
try:
    nltk.download('punkt_tab')
except Exception:
    pass
nltk.download('wordnet')
nltk.download('stopwords')

def sent_tokenize_safe(text, lang="english"):
    try:
        return nltk.sent_tokenize(text, language=lang)
    except LookupError:
        nltk.download('punkt')
        return nltk.sent_tokenize(text, language=lang)

def word_tokenize_safe(text, lang="english"):
    try:
        return nltk.word_tokenize(text, language=lang)
    except LookupError:
        nltk.download('punkt')
        return nltk.word_tokenize(text, language=lang)

# --------------------- 1) Load Gutenberg text ---------------------
GUT_URL = "https://www.gutenberg.org/files/1342/1342-0.txt"
LOCAL_FALLBACK = "1342-0.txt"  # put local copy here if offline

def read_gutenberg(url=GUT_URL, local=LOCAL_FALLBACK) -> str:
    txt = None
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            txt = r.read().decode("utf-8", errors="ignore")
            print("[INFO] Downloaded text from Gutenberg.")
    except Exception as e:
        print(f"[WARN] Online fetch failed: {e}\n[INFO] Trying local file '{local}' ...")
    if txt is None:
        with open(local, "r", encoding="utf-8") as f:
            txt = f.read()
        print("[INFO] Loaded text from local file.")
    return txt

def strip_gutenberg_boilerplate(text: str) -> str:
    start = text.find("*** START OF THIS PROJECT GUTENBERG EBOOK")
    end = text.find("*** END OF THIS PROJECT GUTENBERG EBOOK")
    if start != -1:
        text = text[text.find("\n", start)+1:]
    if end != -1:
        text = text[:end]
    return text

raw_text = read_gutenberg()
raw_text = strip_gutenberg_boilerplate(raw_text)

# --------------------- 2) Function words list ---------------------
GRAM_FUNCTION_WORDS = {
    # Auxiliaries & modals
    "can","could","may","might","must","shall","should","will","would","ought","need","dare",
    "do","does","did","done","doing",
    "be","am","is","are","was","were","been","being",
    "have","has","had","having",
    # Negation / polarity
    "not","n't","never","neither","nor",
    # Determiners / quantifiers / articles
    "the","a","an","this","that","these","those",
    "some","any","no","none","each","every","either","neither",
    "all","both","half","much","many","more","most","few","fewer","fewest","little","less","least",
    "enough","several","various","other","another","such","same",
    # Pronouns
    "i","me","my","mine","myself",
    "you","your","yours","yourself","yourselves",
    "he","him","his","himself",
    "she","her","hers","herself",
    "it","its","itself",
    "we","us","our","ours","ourselves",
    "they","them","their","theirs","themselves",
    "one","oneself",
    "who","whom","whose","whoever","whomever","which","whichever","what","whatever",
    # Prepositions
    "of","to","in","for","on","with","at","by","from","up","about","than","into","through","after","over",
    "between","out","against","during","without","before","under","around","among","across",
    "toward","towards","upon","within","along","behind","beyond","despite","off","onto","via","per",
    # Conjunctions / subordinators
    "and","or","but","nor","so","yet","for",
    "because","although","though","if","unless","since","while","whereas","whether","that","as",
    "once","until","when","whenever","where","wherever","before","after","so that",
    # Expletives
    "there"
}

stop_set = set(stopwords.words("english"))
stop_set |= GRAM_FUNCTION_WORDS

lemmatizer = WordNetLemmatizer()

def normalize_token(tok: str) -> str:
    t = tok.lower()
    t = re.sub(r'[^a-z]+', '', t)   # keep only letters
    return t

def preprocess_sentence(sent_tokens):
    cleaned = []
    for tok in sent_tokens:
        t = normalize_token(tok)
        if not t:
            continue
        if t in stop_set:
            continue
        t = lemmatizer.lemmatize(t)
        t = lemmatizer.lemmatize(t, 'v')
        if t not in stop_set and len(t) > 1:
            cleaned.append(t)
    return cleaned

# --------------------- 3) Apply preprocessing ---------------------
raw_sents = sent_tokenize_safe(raw_text)
sentences = []
for i, s in enumerate(raw_sents, 1):
    toks = word_tokenize_safe(s)
    ps = preprocess_sentence(toks)
    if ps:
        sentences.append(ps)

print(f"[INFO] Preprocessing done. #sentences: {len(sentences)}")
print("[INFO] Example tokens:", sentences[:2])

# --------------------- 4) Export ---------------------
with open("corpus_tokens.txt", "w", encoding="utf-8") as f:
    for sent in sentences:
        f.write(" ".join(sent) + "\n")

print("[INFO] Wrote: corpus_tokens.txt ✅")
