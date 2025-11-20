# ASSIGNMENT-7: TF-IDF, PMI, and Nearest Neighbor Search

This assignment implements three important NLP techniques: TF-IDF vectorization, Pointwise Mutual Information (PMI) calculation, and nearest neighbor search for sentence similarity.

## Objectives
- Implement TF-IDF vectorization for sentence representation
- Calculate PMI scores for word pairs
- Find nearest neighbors using cosine similarity
- Compare sentences across train/validation/test sets

## Files

### Python Scripts
- `tfidf.py` - TF-IDF vectorization implementation
- `pmi.py` - PMI calculation for bigrams
- `nearest_neighbour.py` - Nearest neighbor search within sets
- `code.py` - Main script for finding nearest neighbors across sets

### Input Files
- `inputs/train.txt` - Training sentences
- `inputs/val.txt` - Validation sentences
- `inputs/test.txt` - Test sentences
- `inputs/unigram_model.txt` - Unigram probabilities
- `inputs/bigram_model.txt` - Bigram probabilities

### Output Files
- `outputs/tfidf_output/`:
  - `tfidf_train.npz` - Sparse TF-IDF matrix for training set
  - `tfidf_val.npz` - Sparse TF-IDF matrix for validation set
  - `tfidf_test.npz` - Sparse TF-IDF matrix for test set
  - `vocab.json` - Vocabulary mapping (token → column index)
- `outputs/pmi_val.txt` - PMI scores for validation bigrams
- `outputs/pmi_test.txt` - PMI scores for test bigrams
- `outputs/nearest_neighbors_val.txt` - Nearest neighbors within validation set
- `outputs/nearest_neighbors_test.txt` - Nearest neighbors within test set
- `outputs/nearest_neighbors_val_in_train.txt` - Validation sentences' nearest neighbors in training set
- `outputs/nearest_neighbors_test_in_train.txt` - Test sentences' nearest neighbors in training set

## Key Features

### TF-IDF (Term Frequency-Inverse Document Frequency)
- **Purpose**: Convert sentences to numerical vectors
- **Implementation**: Uses scikit-learn's TfidfVectorizer
- **Training**: IDF learned only from training set
- **Transformation**: Validation and test sets transformed using training IDF
- **Output**: Sparse matrices saved in NPZ format

### PMI (Pointwise Mutual Information)
- **Formula**: `PMI(w1, w2) = log2(P(w1, w2) / (P(w1) * P(w2)))`
- **Interpretation**:
  - PMI > 0: Words co-occur more than expected
  - PMI < 0: Words co-occur less than expected
  - PMI = 0: Independent occurrence
- **Usage**: Identifies strong word associations

### Nearest Neighbor Search
- **Similarity Metric**: Cosine similarity
- **Within-set**: Finds most similar sentences within same set
- **Cross-set**: Finds most similar training sentences for validation/test sentences
- **Batching**: Processes in batches (default 500) for memory efficiency

## Usage

### 1. Generate TF-IDF Vectors
```bash
python tfidf.py
```
This creates TF-IDF matrices for all three sets and saves vocabulary.

### 2. Calculate PMI Scores
```bash
python pmi.py
```
This computes PMI for all bigrams in validation and test sets.

### 3. Find Nearest Neighbors
```bash
# Within-set neighbors
python nearest_neighbour.py

# Cross-set neighbors (val/test → train)
python code.py
```

## Dependencies
- `scikit-learn` - For TfidfVectorizer and cosine similarity
- `scipy` - For sparse matrix operations (save_npz, load_npz)
- `numpy` - For numerical operations
- `pathlib` - For file path handling

## Output Format

### PMI Output
```
word1 word2    PMI_score
```

### Nearest Neighbor Output
```
query_index    neighbor_index    similarity_score    query_sentence    |||    neighbor_sentence
```

## Results
The implementation provides:
- Dense vector representations of sentences via TF-IDF
- Word association scores via PMI
- Similarity rankings between sentences
- Quantitative measures of semantic similarity

These outputs can be used for:
- Information retrieval
- Document clustering
- Semantic similarity analysis
- Query expansion
