# skipgram_tune_and_tsne.py
# ---------------------------------------------------------
# - Load preprocessed corpus (one sentence per line, tokens space-separated)
# - Grid-search Word2Vec Skip-gram over window × dimension
# - Score by mean cosine of top-K neighbors for 10 themed words
# - Retrain best model; export 5-NN per themed word
# - t-SNE visualization of (centers + neighbors) and save PNG
# ---------------------------------------------------------

from gensim.models import Word2Vec
import pandas as pd
import numpy as np
import os
from collections import OrderedDict  # <-- needed for stable unique ordering
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ---------------------- I/O & Config ----------------------

def read_corpus(path: str = "corpus_tokens.txt"):
    """Read the preprocessed corpus produced elsewhere.
    Each line is a sentence, already tokenized by spaces."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cannot find {path}. Make sure preprocessing ran.")
    with open(path, "r", encoding="utf-8") as f:
        return [line.split() for line in f if line.strip()]

sentences = read_corpus()

# Grid to tune
WINDOW_GRID = [2, 5, 10]
DIM_GRID    = [50, 100, 200]

# Word2Vec training knobs
EPOCHS    = 30
MIN_COUNT = 3
WORKERS   = 4
SEED      = 42

# Themed words (centers)
TOP_KEYWORDS = ["rich","poor",
"marriage","love",
"darcy","elizabeth",
"man","woman"]

K_NN = 5  # neighbors to retrieve for each center

# ---------------------- Helpers ----------------------

def l2_normalize(mat: np.ndarray, axis=1, eps=1e-12) -> np.ndarray:
    """Row- or column-normalize matrix to unit L2 norm (for cosine)."""
    return mat / (np.linalg.norm(mat, axis=axis, keepdims=True) + eps)

def cosine_topk(query_vec: np.ndarray,
                all_mat: np.ndarray,
                all_words: list[str],
                k: int = 10):
    """Return top-k (word, cosine) pairs from pre-normalized matrix."""
    # Normalize query vector
    q = query_vec / (np.linalg.norm(query_vec) + 1e-12)
    sims = all_mat @ q

    # Handle very small vocab safely
    k_eff = min(k + 1, len(all_words))  # +1 because the center often ranks first
    if k_eff <= 0:
        return []

    # Fast partial sort then exact sort inside the slice
    idx = np.argpartition(-sims, np.arange(k_eff))[:k_eff]
    idx = idx[np.argsort(-sims[idx])]

    return [(all_words[i], float(sims[i])) for i in idx]

def eval_model(model: Word2Vec) -> float:
    """Score a model by the mean cosine of top-K neighbors across themed centers."""
    vocab = set(model.wv.index_to_key)
    centers = [w for w in TOP_KEYWORDS if w in vocab]
    if not centers:
        return float("nan")

    all_words = list(model.wv.index_to_key)
    all_mat = l2_normalize(model.wv[all_words], axis=1)

    sims = []
    for c in centers:
        neigh = cosine_topk(model.wv[c], all_mat, all_words, k=K_NN)
        # drop self if present
        sims.extend(s for w, s in neigh if w != c)

    return float(np.mean(sims)) if sims else float("nan")

# ---------------------- Grid search ----------------------

results = []
for w in WINDOW_GRID:
    for dim in DIM_GRID:
        print(f"[TRAIN] window={w}, dim={dim}")
        model = Word2Vec(
            sentences,
            vector_size=dim,
            window=w,
            min_count=MIN_COUNT,
            sg=1,                 # Skip-gram
            workers=WORKERS,
            epochs=EPOCHS,
            seed=SEED
        )
        score = eval_model(model)
        results.append({"window": w, "dim": dim, "score": score})

df = pd.DataFrame(results)
print("\n[TUNING RESULTS] (mean neighbor cosine)")
print(df.pivot(index="dim", columns="window", values="score"))

if df["score"].isna().all():
    raise RuntimeError("All scores are NaN. Are your themed words present in the vocab?")

# ---------------------- Select best & retrain ----------------------

best_row = df.loc[df["score"].idxmax()]
BEST_WINDOW = int(best_row["window"])
BEST_DIM    = int(best_row["dim"])
print(f"\n[SELECT] Best -> window={BEST_WINDOW}, dim={BEST_DIM}, score={best_row['score']:.4f}")

best_model = Word2Vec(
    sentences,
    vector_size=BEST_DIM,
    window=BEST_WINDOW,
    min_count=MIN_COUNT,
    sg=1,
    workers=WORKERS,
    epochs=EPOCHS,
    seed=SEED
)

# ---------------------- 5-NN for themed words ----------------------

all_words = list(best_model.wv.index_to_key)
if len(all_words) == 0:
    raise RuntimeError("Best model vocabulary is empty.")

all_mat = l2_normalize(best_model.wv[all_words], axis=1)

rows = []
picked, groups = [], []
centers_present = [w for w in TOP_KEYWORDS if w in best_model.wv.key_to_index]
centers_set = set(centers_present)

for c in centers_present:
    neigh = cosine_topk(best_model.wv[c], all_mat, all_words, k=K_NN)
    neigh = [(w, s) for w, s in neigh if w != c][:K_NN]
    for wword, sim in neigh:
        rows.append({"center": c, "neighbor": wword, "similarity": sim})
    # gather words for plotting
    picked.append(c);  groups.append(c)
    for wword, _ in neigh:
        picked.append(wword);  groups.append(c)

neighbors_df = pd.DataFrame(rows)
print("\n[NEIGHBORS] Top-5 per themed word:")
print(neighbors_df if not neighbors_df.empty else "(no neighbors found)")

os.makedirs("best_model_out", exist_ok=True)
neighbors_df.to_csv("best_model_out/best_neighbors_top5.csv", index=False)
best_model.save(f"best_model_out/w2v_sg_win{BEST_WINDOW}_dim{BEST_DIM}.model")

# ---------------------- t-SNE visualization ----------------------

# If we have nothing to plot, exit gracefully
if not picked:
    print("[WARN] No centers had neighbors; skipping t-SNE.")
else:
    # Stable unique order while preserving first occurrence
    picked_unique = list(OrderedDict((w, None) for w in picked).keys())

    # Map each picked word to its designated center (first mapping wins)
    first_group = {}
    for wword, grp in zip(picked, groups):
        if wword not in first_group:
            first_group[wword] = grp
    groups_unique = [first_group[w] for w in picked_unique]

    # Assemble vectors for plotting
    X = np.vstack([best_model.wv[w] for w in picked_unique]).astype(np.float64)

    # Standardize for t-SNE stability
    X_std = (X - X.mean(axis=0, keepdims=True)) / (X.std(axis=0, keepdims=True) + 1e-12)

    # Perplexity in [5, 30] and < n/3
    n_pts = X_std.shape[0]
    perp = max(5, min(30, max(5, (n_pts - 1) // 3)))

    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        init="pca",
        random_state=42
        # omit learning_rate/n_iter to remain compatible with older sklearn
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

    plt.title(f"Skip-gram t-SNE — best(window={BEST_WINDOW}, dim={BEST_DIM}); "
              f"{len(centers_present)} centers, top-{K_NN} neighbors")
    plt.xlabel("t-SNE 1"); plt.ylabel("t-SNE 2")
    plt.legend(title="Center word", fontsize=9)
    plt.grid(alpha=0.25)
    plt.tight_layout()

    # Save + show
    out_png = f"best_model_out/tsne_win{BEST_WINDOW}_dim{BEST_DIM}.png"
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"[INFO] t-SNE saved to {out_png}")

print("\n[DONE] Training, selection, neighbors, and t-SNE finished.")
