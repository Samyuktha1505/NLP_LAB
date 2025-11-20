# ASSIGNMENT-6: Advanced Smoothing Techniques and Text Generation

This assignment implements advanced smoothing techniques (Kneser-Ney and Katz backoff) for n-gram language models and uses them for text generation with greedy and beam search algorithms.

## Objectives
- Implement Kneser-Ney smoothing for n-gram models
- Implement Katz backoff smoothing
- Generate text using greedy decoding
- Generate text using beam search decoding
- Compare smoothing techniques on quadrigram models

## Files
- `U23AI059_Lab6.ipynb` - Main implementation notebook
- `U23AI059_1ST.ipynb` - Kneser-Ney implementation
- `U23AI059_3RD.ipynb` - Text generation (greedy and beam search)
- `task2.ipynb` - Additional tasks
- `quadrigram_katz.csv` - Katz-smoothed quadrigram probabilities
- `quadrigram_kneserney.csv` - Kneser-Ney smoothed quadrigram probabilities

## Key Features

### Kneser-Ney Smoothing
- **Recursive formulation**: Handles higher-order n-grams with backoff
- **Discount parameter**: Typically d=0.75
- **Continuation probability**: For unigrams, uses number of unique histories
- **Formula**:
  ```
  P_KN(w|context) = max(c(w,context) - d, 0) / c(context) 
                   + λ(context) * P_KN(w|context[1:])
  ```

### Katz Backoff
- Uses Good-Turing for low-frequency n-grams
- Backs off to lower-order models when higher-order n-gram is unseen
- Handles sparse data better than simple interpolation

### Text Generation

#### Greedy Decoding
- At each step, selects the most probable next word
- Simple and fast
- May produce repetitive or low-quality text

#### Beam Search
- Maintains top-k candidates at each step
- Beam size typically 10-20
- Better quality than greedy, more diverse outputs
- Balances quality and diversity

## Usage

### Kneser-Ney Smoothing
```python
# Load n-gram counts
ngram_counts = load_ngram_counts()

# Calculate Kneser-Ney probability
prob = kn_prob(ngram, ngram_counts, d=0.75)
```

### Text Generation
```python
# Greedy generation
sentence = generate_greedy_ng(ngram_counts, n=4)

# Beam search generation
sentences = generate_beam_ng(ngram_counts, n=4, beam_size=20)
```

## Dependencies
- `collections` - For Counter and defaultdict
- `random` - For sampling
- `math` - For probability calculations
- N-gram counts from Assignment 4

## Results
The implementation:
- Generates 100 sentences using greedy decoding for trigram and quadrigram models
- Generates 100 sentences using beam search (beam_size=20)
- Saves results to CSV files for analysis
- Provides comparison between smoothing techniques

## Output Files
- `greedy_3gram_100.csv` - Greedy trigram sentences
- `greedy_4gram_100.csv` - Greedy quadrigram sentences
- `beam_3gram_100.csv` - Beam search trigram sentences
- `beam_4gram_100.csv` - Beam search quadrigram sentences

The generated text demonstrates the language model's ability to produce coherent Telugu sentences based on learned n-gram patterns.
