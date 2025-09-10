# Project Title: Comparative Analysis of Word Embedding Techniques  

This project compares three word embedding algorithms—**Skip-gram (Word2Vec)**, **GloVe**, and **SPPMI–SVD**—on Jane Austen’s *Pride and Prejudice*. It investigates how predictive, count-based, and hybrid approaches capture semantic meaning in a literary text.  

---

## 📁 Folder Contents  
- `corpus.py` – Preprocessing pipeline (cleaning, tokenization, lemmatization)  
- `skip-gram.py` – Implementation of Word2Vec (Skip-gram)  
- `SPPMI-SVD.py` – Count-based embeddings via SPPMI + SVD  
- `GloVe.R` – GloVe implementation (R `text2vec`)  
- `Comparative Analysis on Word Embedding Techniques.pdf` – Full project report  
- `README.md` – Project documentation (this file)  

---

## 📌 Project Overview  

### 📖 Background  
Word embeddings are a foundational representation technique in NLP, mapping words into continuous vector spaces. This project evaluates different embedding paradigms using a classic literary text (*Pride and Prejudice*) to study how models capture semantics such as gender, class, and relationships.  

### 🎯 Objectives  
- Build a unified preprocessing pipeline for text normalization  
- Train three types of embeddings: Word2Vec (Skip-gram), SPPMI–SVD, and GloVe  
- Compare embeddings through semantic neighborhoods and visualization  
- Analyze trade-offs between predictive and count-based approaches  

---

## 📊 Methodology  

### 🔍 Data Preprocessing  
- Source: *Pride and Prejudice* (Project Gutenberg)  
- Normalized whitespace, lowercased text, tokenized sentences/words  
- Inserted `<sos>` and `<eos>` markers to preserve sentence boundaries  
- Vocabulary built from training split only to avoid leakage  

### 🧪 Models Used  
**Skip-gram (Word2Vec):**  
- Predictive neural embeddings trained with negative sampling  
- Captures syntagmatic and semantic associations  

**SPPMI–SVD:**  
- Count-based approach  
- Factorizes shifted Positive PMI co-occurrence matrix  
- Highlights statistical co-occurrence patterns  

**GloVe:**  
- Hybrid model using co-occurrence regression  
- Implemented in R (`text2vec`)  

### 📈 Evaluation  
- **Intrinsic:** cosine similarity neighborhoods for probe words (*love, marriage, rich, poor*)  
- **Visualization:** embeddings projected via t-SNE  
- **Qualitative:** inspected clustering of characters and social concepts  

---

## 🧠 Key Learnings  
- Preprocessing choices strongly affect embedding quality  
- Predictive models (Word2Vec) capture nuanced semantics but require more compute  
- Count-based methods (SPPMI–SVD) are efficient but weaker on rare words  
- GloVe provides a balanced middle ground  
- Visualization (t-SNE) is valuable for comparing embedding spaces  

---

## 📄 Report  
📑 - [Comparative Analysis on Word Embedding Techniques (PDF)](https://github.com/zhijing31/Word-Embeddings-on-Pride-and-Prejudice/blob/main/Comparative%20Analysis%20on%20Word%20Embedding%20Techniques.pdf) 
