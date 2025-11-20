# ASSIGNMENT-9: Subword Tokenization - BPE and WordPiece

This assignment implements two popular subword tokenization algorithms: Byte Pair Encoding (BPE) and WordPiece, for handling Telugu text and reducing vocabulary size while maintaining morphological information.

## Objectives
- Implement Fast BPE (Byte Pair Encoding) algorithm
- Implement WordPiece tokenization algorithm
- Train tokenizers on Telugu corpus
- Compare tokenization strategies
- Handle out-of-vocabulary words effectively

## Files
- `U23AI059_ASSIGNMENT-9.IPYNB` - Main implementation notebook

## Key Features

### Byte Pair Encoding (BPE)
- **Algorithm**:
  1. Initialize vocabulary with all characters
  2. Count frequency of all character pairs
  3. Iteratively merge most frequent pairs
  4. Continue until desired vocabulary size (default: 32,000 merges)

- **Key Components**:
  - `FastBPE` class with optimized merge operations
  - Tracks word frequencies and pair statistics
  - Efficient merge operation using word instances
  - Special token: `</w>` marks word boundaries

- **Encoding**:
  - Applies learned merges to tokenize new words
  - Handles unseen words by applying available merges

### WordPiece
- **Algorithm**:
  1. Initialize with character vocabulary
  2. Tokenize words using longest-match strategy
  3. Count pair frequencies in current tokenization
  4. Merge most frequent pairs
  5. Repeat until vocabulary size reached

- **Key Components**:
  - `WordPiece` class with greedy tokenization
  - Uses `##` prefix for subword tokens (not at word start)
  - `[UNK]` token for out-of-vocabulary words
  - Longest-match tokenization strategy

- **Tokenization**:
  - Greedily matches longest possible subword from vocabulary
  - Returns `[UNK]` if word cannot be tokenized

## Usage

### Training BPE
```python
from collections import Counter, defaultdict

bpe = FastBPE()
bpe.train(corpus, num_merges=32000)
tokens = bpe.encode("నేను మంచి పిల్లను")
```

### Training WordPiece
```python
wp = WordPiece()
wp.train(corpus, vocab_size=32000)
tokens = wp.tokenize("నేను మంచి పిల్లను")
```

## Algorithm Comparison

| Feature | BPE | WordPiece |
|---------|-----|-----------|
| Merge Strategy | Most frequent pairs | Most frequent pairs (with likelihood) |
| Tokenization | Apply merges in order | Longest-match greedy |
| OOV Handling | Applies available merges | Returns [UNK] |
| Special Tokens | `</w>` for word boundary | `##` for subwords, `[UNK]` for unknown |
| Training Speed | Fast (optimized) | Moderate |

## Dependencies
- `collections` - For Counter and defaultdict
- Standard Python libraries

## Results
The implementation demonstrates:
- **BPE**: Successfully learns subword units from Telugu corpus
- **WordPiece**: Creates vocabulary with morphological awareness
- Both methods reduce vocabulary size while preserving linguistic information
- Effective handling of Telugu script and morphological variations

## Applications
- Machine translation
- Language modeling
- Preprocessing for transformer models
- Handling morphologically rich languages like Telugu

## Advantages
1. **Reduced Vocabulary**: Subword units reduce vocabulary size
2. **OOV Handling**: Better handling of rare/unseen words
3. **Morphological Awareness**: Captures word structure
4. **Language Agnostic**: Works well for various languages including Telugu
