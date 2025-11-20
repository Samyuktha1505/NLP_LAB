# ASSIGNMENT-5: Dataset Splitting and Language Model Evaluation

This assignment focuses on preparing datasets for language model evaluation by splitting data into train/validation/test sets and implementing perplexity calculation.

## Objectives
- Split Telugu dataset into train, validation, and test sets
- Load pre-trained n-gram models from Assignment 4
- Calculate sentence probabilities using Good-Turing smoothed models
- Compute perplexity scores for language model evaluation
- Compare performance across different n-gram orders

## Files
- `Q1.ipynb` - Question 1: Dataset splitting
- `Q2.ipynb` - Question 2: Language model evaluation and perplexity
- `train.txt` - Training set (23,001 sentences)
- `validation.txt` - Validation set (1,000 sentences)
- `test.txt` - Test set (1,000 sentences)
- `validation.csv` - Validation set in CSV format
- `test.csv` - Test set in CSV format

## Key Features

### Dataset Splitting (Q1)
- Random shuffle with seed=42 for reproducibility
- Split ratio:
  - Training: 23,001 sentences (92%)
  - Validation: 1,000 sentences (4%)
  - Test: 1,000 sentences (4%)
- Saves both TXT and CSV formats

### Language Model Evaluation (Q2)
- **Sentence Probability**: Calculates log probability of sentences
  - Uses Good-Turing smoothed n-gram models
  - Handles unseen n-grams with P_unseen
  - Works in log-space to avoid underflow

- **Perplexity Calculation**:
  - Formula: `PP = exp(-log_prob / length)`
  - Lower perplexity = better model
  - Measures how "surprised" the model is by the test data

### Model Comparison
- Evaluates unigram, bigram, trigram, and quadrigram models
- Compares perplexity scores across different n-gram orders
- Typically, higher-order n-grams perform better (lower perplexity)

## Usage

### Q1: Dataset Splitting
```python
# Load dataset
sentences = load_sentences("telugu_dataset.txt")

# Split with seed
train, val, test = split_dataset(sentences, seed=42)

# Save to files
save_splits(train, val, test)
```

### Q2: Evaluation
```python
# Load models
models = load_ngram_models()

# Calculate perplexity
perplexity = calculate_perplexity(sentence, model, n)
```

## Dependencies
- `pandas` - For CSV handling
- `numpy` - For numerical operations
- `math` - For log calculations
- Models from Assignment 4 (CSV files)

## Results
The evaluation provides:
- Perplexity scores for each n-gram model
- Comparison of model performance
- Quantitative assessment of language model quality

Lower perplexity indicates the model is better at predicting the test data, with higher-order n-grams generally performing better than lower-order models.
