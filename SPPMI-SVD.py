#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 26 21:05:21 2025

@author: chenzhijing
"""

# sppmi_svd_tune_and_tsne.py
# ---------------------------------------------------------
# SPPMI+SVD ONLY (no preprocessing, no word2vec)
# - Reads preprocessed corpus: one sentence per line, tokens space-separated
# - Builds vocab (by min freq & max vocab cap)
# - Grid-search hyperparams: WINDOW_SIZE × SHIFT (k) × RANK (embedding dim)
# - Score = mean cosine of top-K neighbors for 10 themed words
# - Recompute best SPPMI+SVD, export 5-NN per themed word
# - t-SNE visualization of (centers + neighbors) and save PNG
# ---------------------------------------------------------

import os
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE

# ---------------------- I/O & Config ----------------------

def read_corpus(path: str = "corpus_tokens.txt"):
    """Read the preprocessed corpus produced elsewhere.
    Each line is a sentence, already tokenized by spaces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Make sure preprocessing ran.")
    with open(path, "r", encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]

sentences = read_corpus()
print(f"[INFO] Loaded corpus with {len(sentences)} sentences.")

# Themed words (centers)
TOP_KEYWORDS = [
    "rich","poor",
    "marriage","love",
    "darcy","elizabeth",
    "man","woman"
]
K_NN = 5  # neighbors to retrieve per center

# Vocab building knobs
MIN_FREQ   = 5         # drop rarer tokens
MAX_VOCAB  = 2000      # cap vocabulary to keep SVD tractable (VxV matrix)

# Tuning grids
WINDOW_GRID = [2, 3, 5]        # co-occurrence window size (symmetric)
SHIFT_GRID  = [5.0, 10.0]      # SPPMI shift parameter k (larger → more sparsity)
RANK_GRID   = [50, 100]        # SVD embedding dimension

OUT_DIR = "sppmi_out"
os.makedirs(OUT_DIR, exist_ok=True)

# ---------------------- Helpers ----------------------

def l2_normalize(mat: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    """Row/column normalize matrix to unit L2 norm (for cosine)."""
    return mat / (np.linalg.norm(mat, axis=axis, keepdims=True) + eps)

def cosine_topk(query_vec: np.ndarray,
                all_mat: np.ndarray,
                all_words: list[str],
                k: int = 10):
    """Return top-k (word, cosine) pairs from pre-normalized matrix."""
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = all_mat @ q
    k_eff = min(k + 1, len(all_words))  # +1 because self often at rank1
    if k_eff <= 0:
        return []
    idx = np.argpartition(-sims, np.arange(k_eff))[:k_eff]
    idx = idx[np.argsort(-sims[idx])]
    return [(all_words[i], float(sims[i])) for i in idx]

def build_vocab(sentences, min_freq=5, max_vocab=2000):
    """Return (vocab dict, vocab_words list, freq Counter)."""
    freq = Counter(w for s in sentences for w in s)
    vocab_words = [w for w, c in freq.most_common(max_vocab) if c >= min_freq]
    vocab = {w: i for i, w in enumerate(vocab_words)}
    return vocab, vocab_words, freq

def build_cooccurrence(sentences, vocab: dict[str,int], window: int) -> np.ndarray:
    """Dense co-occurrence matrix (V×V) within symmetric window."""
    V = len(vocab)
    co = np.zeros((V, V), dtype=np.float32)
    for sent in sentences:
        idxs = [vocab[w] for w in sent if w in vocab]
        n = len(idxs)
        for i, wi in enumerate(idxs):
            start = max(0, i - window)
            end = min(n, i + window + 1)
            for j in range(start, end):
                if j == i: 
                    continue
                wj = idxs[j]
                co[wi, wj] += 1.0
    return co

def sppmi_from_co(co: np.ndarray, shift_k: float) -> np.ndarray:
    """Compute SPPMI matrix from a co-occurrence matrix."""
    co_sum = float(co.sum())
    if co_sum == 0.0:
        raise ValueError("[SSPMI] Co-occurrence matrix is all zeros; adjust filters/window.")
    p_word = co.sum(axis=1) / co_sum

    # Only compute on non-zeros of co to save time
    sppmi = np.zeros_like(co, dtype=np.float32)
    eps = 1e-12
    for i in range(co.shape[0]):
        nz = np.nonzero(co[i])[0]
        if nz.size == 0:
            continue
        cij = co[i, nz] / co_sum
        # Avoid log(0) with eps; PMI = log(p(i,j)/(p(i)p(j)))
        pmi = np.log((cij + eps) / (p_word[i] * p_word[nz] + eps))
        val = pmi - np.log(shift_k)
        sppmi[i, nz] = np.maximum(val, 0.0)
    return sppmi

def svd_embed(sppmi: np.ndarray, rank: int) -> np.ndarray:
    """SVD on SPPMI; return embeddings U * sqrt(S) of given rank."""
    U, S, _ = np.linalg.svd(sppmi, full_matrices=False)
    r = int(min(rank, S.size))
    return (U[:, :r] @ np.diag(np.sqrt(S[:r]))).astype(np.float32)

def eval_embeddings(emb: np.ndarray, vocab_words: list[str]) -> float:
    """Score = mean cosine of top-K neighbors for TOP_KEYWORDS."""
    word2idx = {w: i for i, w in enumerate(vocab_words)}
    centers = [w for w in TOP_KEYWORDS if w in word2idx]
    if not centers:
        return float("nan")
    E = l2_normalize(emb, axis=1)
    sims = []
    for c in centers:
        neigh = cosine_topk(E[word2idx[c]], E, vocab_words, k=K_NN)
        sims.extend(s for w, s in neigh if w != c)
    return float(np.mean(sims)) if sims else float("nan")

# ---------------------- Build vocab once ----------------------

vocab, vocab_words, freq = build_vocab(sentences, MIN_FREQ, MAX_VOCAB)
V = len(vocab)
print(f"[INFO] Vocab size after pruning: {V} (min_freq={MIN_FREQ}, max_vocab={MAX_VOCAB})")
if V == 0:
    raise RuntimeError("Empty vocabulary. Lower MIN_FREQ or raise MAX_VOCAB.")

# ---------------------- Grid search ----------------------

records = []

for win in WINDOW_GRID:
    print(f"\n[STAGE] Building co-occurrence for window={win} ...")
    co = build_cooccurrence(sentences, vocab, window=win)

    for shift in SHIFT_GRID:
        # SPPMI depends on 'shift'
        sppmi = sppmi_from_co(co, shift_k=shift)

        for rank in RANK_GRID:
            print(f"[EVAL] window={win}, shift={shift}, rank={rank}")
            emb = svd_embed(sppmi, rank=rank)
            score = eval_embeddings(emb, vocab_words)
            records.append({
                "window": win,
                "shift": shift,
                "rank": rank,
                "score": score
            })

df = pd.DataFrame(records)
print("\n[TUNING RESULTS] (mean neighbor cosine)")
print(df.pivot_table(index=["rank"], columns=["window","shift"], values="score"))

if df["score"].isna().all():
    raise RuntimeError("All scores are NaN. Are your themed words present in the vocab?")

best = df.loc[df["score"].idxmax()]
BEST_WINDOW = int(best["window"])
BEST_SHIFT  = float(best["shift"])
BEST_RANK   = int(best["rank"])
print(f"\n[SELECT] Best -> window={BEST_WINDOW}, shift={BEST_SHIFT}, rank={BEST_RANK}, score={best['score']:.4f}")

# ---------------------- Recompute best model ----------------------

print(f"\n[REBUILD] Best co-occurrence (window={BEST_WINDOW})")
co_best = build_cooccurrence(sentences, vocab, window=BEST_WINDOW)
print(f"[REBUILD] Best SPPMI (shift={BEST_SHIFT})")
sppmi_best = sppmi_from_co(co_best, shift_k=BEST_SHIFT)
print(f"[REBUILD] Best SVD (rank={BEST_RANK})")
emb_best = svd_embed(sppmi_best, rank=BEST_RANK)

# ---------------------- 5-NN for themed words ----------------------

word2idx = {w: i for i, w in enumerate(vocab_words)}
centers_present = [w for w in TOP_KEYWORDS if w in word2idx]
centers_set = set(centers_present)

E = l2_normalize(emb_best, axis=1)
rows = []
picked, groups = [], []

for c in centers_present:
    neigh = cosine_topk(E[word2idx[c]], E, vocab_words, k=K_NN)
    neigh = [(w, s) for w, s in neigh if w != c][:K_NN]
    for w, s in neigh:
        rows.append({"center": c, "neighbor": w, "similarity": s})
    picked.append(c);  groups.append(c)
    for w, _ in neigh:
        picked.append(w);  groups.append(c)

neighbors_df = pd.DataFrame(rows)
print("\n[NEIGHBORS] Top-5 per themed word:")
print(neighbors_df if not neighbors_df.empty else "(no neighbors found)")

neighbors_df.to_csv(os.path.join(OUT_DIR, "sppmi_best_neighbors_top5.csv"), index=False)
np.save(os.path.join(OUT_DIR, f"sppmi_embedding_win{BEST_WINDOW}_shift{BEST_SHIFT}_rank{BEST_RANK}.npy"), emb_best)
with open(os.path.join(OUT_DIR, "vocab_words.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(vocab_words))

# ---------------------- t-SNE visualization ----------------------

if not picked:
    print("[WARN] No centers had neighbors; skipping t-SNE.")
else:
    # Stable unique order while preserving first occurrence
    picked_unique = list(OrderedDict((w, None) for w in picked).keys())

    # Map each picked word to its center
    first_group = {}
    for w, g in zip(picked, groups):
        if w not in first_group:
            first_group[w] = g
    groups_unique = [first_group[w] for w in picked_unique]

    # Build matrix for the picked words
    X = np.vstack([E[word2idx[w]] for w in picked_unique]).astype(np.float64)

    # Standardize for stability
    X_std = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

    # Perplexity in [5, 30] and < n/3
    n_pts = X_std.shape[0]
    perp = max(5, min(30, max(5, (n_pts - 1) // 3)))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        random_state=42
        # omit learning_rate/n_iter to keep broad sklearn compatibility
    )
    coords = tsne.fit_transform(X_std)

    # Plot
    uniq_groups = list(OrderedDict((g, None) for g in groups_unique).keys())
    cmap = cm.get_cmap('tab10', len(uniq_groups))

    plt.figure(figsize=(12, 8))

    # 1) neighbors
    for gi, g in enumerate(uniq_groups):
        color = cmap(gi)
        idxs = [i for i,(lab,grp) in enumerate(zip(picked_unique, groups_unique))
                if grp == g and lab not in centers_set]
        if idxs:
            plt.scatter(coords[idxs,0], coords[idxs,1], s=30, alpha=0.9, label=g, color=color)
            for i in idxs:
                plt.annotate(picked_unique[i], (coords[i,0], coords[i,1]),
                             fontsize=9, alpha=0.9)

    # 2) centers (big X markers)
    for gi, g in enumerate(uniq_groups):
        color = cmap(gi)
        idxs_c = [i for i,(lab,grp) in enumerate(zip(picked_unique, groups_unique))
                  if grp == g and lab in centers_set]
        if idxs_c:
            plt.scatter(coords[idxs_c,0], coords[idxs_c,1], s=140, marker='X',
                        edgecolor='black', linewidths=0.6, color=color)
            for i in idxs_c:
                plt.annotate(picked_unique[i], (coords[i,0], coords[i,1]),
                             fontsize=11, fontweight='bold')

    plt.title(f"SPPMI+SVD t-SNE — best(win={BEST_WINDOW}, shift={BEST_SHIFT}, rank={BEST_RANK}); "
              f"{len(centers_present)} centers, top-{K_NN} neighbors")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="Center word", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    out_png = os.path.join(OUT_DIR, f"tsne_win{BEST_WINDOW}_shift{BEST_SHIFT}_rank{BEST_RANK}.png")
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"[INFO] t-SNE saved to {out_png}")

print("\n[DONE] SPPMI+SVD tuning, selection, neighbors, and t-SNE finished.")
