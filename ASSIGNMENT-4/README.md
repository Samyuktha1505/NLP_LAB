# ASSIGNMENT-4: N-gram Language Models with Good-Turing Smoothing

This assignment implements n-gram language models (unigram, bigram, trigram, and quadrigram) for Telugu text with Good-Turing smoothing to handle unseen n-grams.

## Objectives
- Build n-gram language models (1-gram to 4-gram)
- Implement Good-Turing smoothing for probability estimation
- Calculate n-gram counts and probabilities
- Save models to CSV files for future use

## Files
- `U23AI059_Lab4_Q1.ipynb` - Question 1: N-gram counting and model building
- `U23AI059_Lab4_Q2.ipynb` - Question 2: Additional analysis
- `U23AI059_Lab4_Q3.ipynb` - Question 3: Extended functionality
- `N-GRAMS.ipynb` - Complete n-gram implementation
- `unigram.csv` - Unigram counts and probabilities
- `bigram.csv` - Bigram counts and probabilities
- `trigram.csv` - Trigram counts and probabilities
- `quadrigram.csv` - Quadrigram counts and probabilities

## Key Features

### N-gram Counting
- Uses sentence and word tokenizers from Assignment 1
- Adds padding tokens: `<s>` (start) and `</s>` (end)
- Counts all n-grams from 1 to 4

### Good-Turing Smoothing
- Handles unseen n-grams by estimating their probability
- Uses frequency of frequencies (Nc) to adjust counts
- Formula: `P_unseen = (N1 / N) / unseen_count`
- For seen n-grams: `P(ngram) = count / total`

### Model Structure
- **Unigram Model**: P(word)
- **Bigram Model**: P(word2 | word1)
- **Trigram Model**: P(word3 | word1, word2)
- **Quadrigram Model**: P(word4 | word1, word2, word3)

## Usage
1. Load Telugu dataset from Assignment 1
2. Tokenize sentences and words
3. Count n-grams with appropriate padding
4. Apply Good-Turing smoothing
5. Save models to CSV files

## Dependencies
- `nltk` - For n-gram generation (`nltk.util.ngrams`)
- `collections` - For Counter and defaultdict
- `re` - For tokenization

## Results
The implementation generates comprehensive n-gram models with:
- Unigrams: ~60,000+ unique tokens
- Bigrams: ~230,000+ unique pairs
- Trigrams: ~300,000+ unique triplets
- Quadrigrams: ~320,000+ unique quadruplets

These models can be used for language modeling, text generation, and probability estimation tasks.
