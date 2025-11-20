# ASSIGNMENT-10: Hidden Markov Model (HMM) for Part-of-Speech Tagging

This assignment implements a Hidden Markov Model (HMM) for Part-of-Speech (POS) tagging using the WSJ (Wall Street Journal) tagged corpus. The implementation includes k-fold cross-validation for model evaluation.

## Objectives
- Implement HMM-based POS tagger
- Train emission and transition probabilities
- Perform k-fold cross-validation (k=5)
- Evaluate tagging accuracy
- Handle unknown words with smoothing

## Files
- `U23AI059_ASSINGMENT-10.ipynb` - Main HMM POS tagger implementation
- `wsj_pos_tagged_en.txt` - Wall Street Journal POS-tagged corpus

## Key Features

### HMM Tagger Class
- **Emission Probabilities**: P(word | tag)
  - Probability of observing a word given its POS tag
  - Uses Laplace/add-one smoothing for unknown words

- **Transition Probabilities**: P(tag_i | tag_{i-1})
  - Probability of tag sequence
  - Models tag dependencies

- **Smoothing**: 
  - Add-one (Laplace) smoothing for emission probabilities
  - Prevents zero probabilities for unseen words

### K-Fold Cross-Validation
- **Purpose**: Robust evaluation without data leakage
- **Process**:
  1. Shuffle data with fixed seed (42)
  2. Split into k=5 folds
  3. For each fold:
     - Train on 4 folds
     - Test on 1 fold
     - Calculate accuracy
  4. Average accuracy across all folds

### Viterbi Algorithm (Implied)
- Decoding algorithm for HMM
- Finds most likely tag sequence given word sequence
- Uses dynamic programming for efficiency

## Data Format
The WSJ corpus uses format: `word/tag`
- Example: `The_DT quick_JJ brown_JJ fox_NN`
- Tags follow Penn Treebank tagset

## Usage

### Loading Tagged Corpus
```python
def load_tagged_corpus(path="wsj_pos_tagged_en.txt"):
    sentences = []
    # Parse word/tag format
    return sentences
```

### Training HMM
```python
tagger = HMMTagger(smoothing=1.0)
tagger.train(training_sentences)
```

### K-Fold Evaluation
```python
folds = k_fold_split(data, k=5, seed=42)
for i, (train_fold, test_fold) in enumerate(folds):
    tagger.train(train_fold)
    accuracy = tagger.evaluate(test_fold)
```

## Dependencies
- `collections` - For defaultdict and Counter
- `random` - For k-fold splitting
- `math` - For probability calculations

## Evaluation Metrics
- **Accuracy**: Percentage of correctly tagged words
- **Per-fold Accuracy**: Individual fold performance
- **Average Accuracy**: Mean across all k folds

## Results
The HMM tagger:
- Learns POS tag patterns from WSJ corpus
- Handles unknown words through smoothing
- Provides robust evaluation via k-fold cross-validation
- Demonstrates effectiveness of probabilistic sequence models

## Tagset
Uses Penn Treebank POS tagset, including:
- **NN**: Noun
- **VB**: Verb
- **JJ**: Adjective
- **DT**: Determiner
- **IN**: Preposition
- And many more fine-grained tags

## Applications
- Text preprocessing for NLP pipelines
- Syntactic parsing
- Information extraction
- Named entity recognition (as preprocessing step)

## Advantages of HMM
1. **Probabilistic**: Handles uncertainty
2. **Sequence Modeling**: Captures tag dependencies
3. **Efficient**: Viterbi algorithm is O(n×m²) where n=words, m=tags
4. **Interpretable**: Clear probabilistic model
