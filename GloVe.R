library(text2vec)
library(ggplot2)
library(Rtsne)
library(doParallel)
library(foreach)
library(purrr)

# ========= 1) Load Python-exported corpus =========
corpus_path <- "corpus_tokens.txt"  # <- set path if needed
sentences <- readLines(corpus_path, encoding = "UTF-8")

# Tokenizer that respects the Python export (space-split)
space_tokenizer <- function(x) strsplit(x, "\\s+")
tokens <- space_tokenizer(sentences)

# ========= 2) Hyperparameter Tuning Setup =========
TOP_KEYWORDS <- tolower(c(
  "rich","poor",
  "marriage","love",
  "darcy","elizabeth",
  "man","woman"
))

# Define hyperparameter grids
glove_params <- expand.grid(
  rank = c(50, 100, 150),
  x_max = c(5, 10, 15),
  window_size = c(3, 5, 8),
  n_iter = c(30, 50, 100)
)

tsne_params <- expand.grid(
  perplexity = c(5, 10, 15, 20),
  max_iter = c(500, 1000, 1500),
  learning_rate = c(100, 200, 500)
)

# ========= 3) GloVe Hyperparameter Tuning Function =========
tune_glove <- function(tokens, params, n_folds = 3) {
  
  # Create vocabulary
  it <- itoken(tokens, progressbar = FALSE)
  vocab <- create_vocabulary(it)
  vocab <- prune_vocabulary(vocab, term_count_min = 1)
  
  results <- list()
  
  for (i in 1:nrow(params)) {
    cat(sprintf("Testing GloVe params %d/%d: rank=%d, x_max=%d, window=%d, iter=%d\n",
                i, nrow(params), params$rank[i], params$x_max[i], 
                params$window_size[i], params$n_iter[i]))
    
    tryCatch({
      # Create TCM with current window size
      it_tcm <- itoken(tokens, progressbar = FALSE)
      tcm <- create_tcm(it_tcm, vectorizer = vocab_vectorizer(vocab), 
                        skip_grams_window = params$window_size[i])
      
      # Train GloVe
      glove <- GlobalVectors$new(rank = params$rank[i], x_max = params$x_max[i])
      word_vectors <- glove$fit_transform(tcm, n_iter = params$n_iter[i], 
                                          convergence_tol = 0.01)
      word_vectors <- word_vectors + t(glove$components)
      
      # Evaluate embedding quality
      eval_score <- evaluate_embeddings(word_vectors, TOP_KEYWORDS)
      
      results[[i]] <- list(
        params = params[i, ],
        score = eval_score,
        embeddings = word_vectors
      )
      
    }, error = function(e) {
      cat(sprintf("Error with params %d: %s\n", i, e$message))
      results[[i]] <- list(params = params[i, ], score = -Inf)
    })
  }
  
  return(results)
}

# ========= 4) Embedding Evaluation Function =========
evaluate_embeddings <- function(word_vectors, keywords) {
  # Normalize embeddings for cosine similarity
  row_norms <- sqrt(rowSums(word_vectors * word_vectors))
  row_norms[row_norms == 0] <- 1e-8
  E <- word_vectors / row_norms
  
  # Keep only keywords present in embeddings
  valid_keywords <- keywords[keywords %in% rownames(E)]
  if (length(valid_keywords) < 2) return(-Inf)
  
  # Calculate average cosine similarity between keyword pairs
  keyword_vectors <- E[valid_keywords, , drop = FALSE]
  similarity_matrix <- keyword_vectors %*% t(keyword_vectors)
  
  # Exclude diagonal (self-similarity)
  diag(similarity_matrix) <- NA
  avg_similarity <- mean(similarity_matrix, na.rm = TRUE)
  
  return(avg_similarity)
}

# ========= 5) Run GloVe Tuning =========
cat("Starting GloVe hyperparameter tuning...\n")
glove_results <- tune_glove(tokens, glove_params)

# Find best GloVe parameters
glove_scores <- map_dbl(glove_results, ~ ifelse(is.null(.x$score), -Inf, .x$score))
best_glove_idx <- which.max(glove_scores)
best_glove_params <- glove_results[[best_glove_idx]]$params
best_embeddings <- glove_results[[best_glove_idx]]$embeddings

cat(sprintf("\nBest GloVe parameters: rank=%d, x_max=%d, window=%d, iter=%d (score: %.4f)\n",
            best_glove_params$rank, best_glove_params$x_max, 
            best_glove_params$window_size, best_glove_params$n_iter,
            glove_scores[best_glove_idx]))

# ========= 6) Get Nearest Neighbors with Best Embeddings =========
row_norms <- sqrt(rowSums(best_embeddings * best_embeddings))
row_norms[row_norms == 0] <- 1e-8
E <- best_embeddings / row_norms

centers <- TOP_KEYWORDS[TOP_KEYWORDS %in% rownames(E)]
cat("[Centers] Available keywords:", paste(centers, collapse = ", "), "\n")

nearest_neighbors <- function(word, E, k = 5) {
  if (!(word %in% rownames(E))) {
    return(data.frame(word = word, neighbor = NA, similarity = NA))
  }
  target <- E[word, , drop = FALSE]
  sims <- as.vector(E %*% as.numeric(target))
  names(sims) <- rownames(E)
  sims <- sort(sims, decreasing = TRUE)
  sims <- sims[names(sims) != word]
  topk <- head(sims, k)
  data.frame(word = word, neighbor = names(topk), similarity = as.numeric(topk))
}

results_kw <- do.call(rbind, lapply(centers, nearest_neighbors, E = E, k = 5))
print(results_kw)

# ========= 7) t-SNE Hyperparameter Tuning =========
tune_tsne <- function(embeddings, params, words_to_plot) {
  emb_sub <- embeddings[words_to_plot, , drop = FALSE]
  
  results <- list()
  
  for (i in 1:nrow(params)) {
    cat(sprintf("Testing t-SNE params %d/%d: perplexity=%d, max_iter=%d, lr=%d\n",
                i, nrow(params), params$perplexity[i], 
                params$max_iter[i], params$learning_rate[i]))
    
    tryCatch({
      set.seed(42)
      tsne_result <- Rtsne(
        emb_sub, 
        perplexity = min(params$perplexity[i], (nrow(emb_sub)-1)/3),
        max_iter = params$max_iter[i],
        learning_rate = params$learning_rate[i],
        verbose = FALSE
      )
      
      # Calculate t-SNE quality metric (lower stress is better)
      stress <- tsne_result$itercosts[length(tsne_result$itercosts)]
      
      results[[i]] <- list(
        params = params[i, ],
        stress = stress,
        tsne_result = tsne_result
      )
      
    }, error = function(e) {
      cat(sprintf("Error with t-SNE params %d: %s\n", i, e$message))
      results[[i]] <- list(params = params[i, ], stress = Inf)
    })
  }
  
  return(results)
}

# Prepare words for t-SNE
edges <- results_kw[, c("word", "neighbor")]
picked <- unique(c(centers, edges$neighbor))
picked <- picked[picked %in% rownames(best_embeddings)]

cat("Starting t-SNE hyperparameter tuning...\n")
tsne_results <- tune_tsne(best_embeddings, tsne_params, picked)

# Find best t-SNE parameters
tsne_stresses <- map_dbl(tsne_results, ~ ifelse(is.null(.x$stress), Inf, .x$stress))
best_tsne_idx <- which.min(tsne_stresses)
best_tsne_params <- tsne_results[[best_tsne_idx]]$params
best_tsne <- tsne_results[[best_tsne_idx]]$tsne_result

cat(sprintf("\nBest t-SNE parameters: perplexity=%d, max_iter=%d, lr=%d (stress: %.4f)\n",
            best_tsne_params$perplexity, best_tsne_params$max_iter, 
            best_tsne_params$learning_rate, tsne_stresses[best_tsne_idx]))

# ========= 8) Final Visualization with Best Parameters =========
df_plot <- data.frame(best_tsne$Y)
colnames(df_plot) <- c("tSNE1", "tSNE2")
df_plot$word <- picked

# Map groups
neigh2center <- results_kw[!duplicated(results_kw$neighbor), c("word", "neighbor")]
colnames(neigh2center) <- c("center", "neighbor")
group <- ifelse(
  df_plot$word %in% centers,
  df_plot$word,
  neigh2center$center[match(df_plot$word, neigh2center$neighbor)]
)
df_plot$group <- factor(group, levels = centers)
df_plot$is_center <- df_plot$word %in% centers

# Create final plot
p <- ggplot(df_plot, aes(x = tSNE1, y = tSNE2, color = group)) +
  geom_point(data = subset(df_plot, !is_center), size = 3, alpha = 0.8) +
  geom_point(data = subset(df_plot, is_center), size = 6, shape = 4, stroke = 1.5) +
  ggrepel::geom_text_repel(aes(label = word), size = 3, show.legend = FALSE, max.overlaps = 20) +
  theme_minimal() +
  labs(
    title = sprintf("Optimized GloVe + t-SNE (rank=%d, window=%d, perplexity=%d)",
                    best_glove_params$rank, best_glove_params$window_size, 
                    best_tsne_params$perplexity),
    subtitle = paste("Centers:", paste(centers, collapse = ", ")),
    x = "t-SNE Dimension 1", y = "t-SNE Dimension 2", color = "Center word"
  ) +
  theme(legend.position = "bottom")

print(p)

# Save results
ggsave("optimized_glove_tsne.png", p, width = 14, height = 10, dpi = 300)
write.csv(results_kw, "optimized_nearest_neighbors.csv", row.names = FALSE)

# ========= 9) Parameter Summary =========
cat("\n=== HYPERPARAMETER TUNING SUMMARY ===\n")
cat("Best GloVe Parameters:\n")
print(best_glove_params)
cat(sprintf("Evaluation Score: %.4f\n", glove_scores[best_glove_idx]))

cat("\nBest t-SNE Parameters:\n")
print(best_tsne_params)
cat(sprintf("Final Stress: %.4f\n", tsne_stresses[best_tsne_idx]))