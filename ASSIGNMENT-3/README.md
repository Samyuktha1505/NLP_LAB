# ASSIGNMENT-3: Trie-Based Stemming for English Nouns

This assignment implements a Trie data structure to perform stemming analysis on English nouns, finding optimal split points between stems and suffixes.

## Objectives
- Implement a Trie data structure for efficient word storage and retrieval
- Find maximum branching points in the Trie to identify stem-suffix boundaries
- Analyze Brown corpus nouns to extract stems and suffixes
- Generate morphological analysis output

## Files
- `U23AI059_Lab3_Q1.ipynb` - Question 1: Trie implementation and stemmer
- `U23AI059_Lab3_Q2.ipynb` - Question 2: Additional analysis
- `brown_nouns.txt` - Brown corpus noun data
- `fst.png` - FST visualization (if applicable)
- `noun_analysis_output.txt` - Stemming analysis results
- `sample.py`, `sample1.py` - Sample implementations

## Key Features

### TrieNode Class
- `children`: Dictionary mapping characters to child nodes
- `count`: Frequency count of words passing through this node
- `end_of_word`: Boolean flag indicating word completion

### Trie Class
- `insert(word)`: Inserts a word into the Trie, updating counts
- `find_split_point(word)`: Finds the optimal point where branching is maximum
  - Returns stem and suffix
  - Maximum branching indicates where multiple words diverge

### Stemming Algorithm
1. Build Trie from all nouns in corpus
2. For each word, traverse Trie and track branching points
3. Select split point with maximum branching (most common divergence point)
4. Extract stem (prefix) and suffix (remainder)

## Usage
1. Load Brown corpus nouns
2. Build Trie structure from all words
3. For each word, find split point
4. Generate output with stem and suffix analysis

## Dependencies
- Standard Python libraries (collections, etc.)

## Results
The Trie-based approach successfully identifies stem-suffix boundaries in English nouns, providing morphological analysis that can be used for further NLP tasks.
