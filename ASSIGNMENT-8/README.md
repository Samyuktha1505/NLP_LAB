## Assignment 8 – NLP Pipelines

### Contents
- `Lab-8.ipynb` – exploration of three classical NLP pipelines:
  - custom preprocessing + TF-IDF weighting
  - WordPiece vocabulary construction and tokenization
  - classifying message intent with smoothed n-gram language models

### Setup
1. Create a Python 3.10+ environment.
2. Install dependencies (only standard lib is required, but `ipykernel` helps for Notebook work):
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install ipykernel
   ```

### Running the Notebook
1. Open `Lab-8.ipynb` in VS Code, Jupyter Lab, or `jupyter notebook`.
2. Execute cells in order. Key checkpoints:
   - **TF-IDF demo**: validates token normalization and weighting.
   - **WordPiece loop**: builds the subword vocabulary; inspect printed vocab.
   - **Intent classifier**: ensures Add-K smoothing is applied and prints the predicted class.

### Expected Outputs
- Preprocessed sentences with placeholders (`<URL>`, `<NUMBER>`, `<PUNCT>`).
- Final WordPiece vocabulary list and tokenization example.
- Probabilities for each intent category and the selected label, e.g.:
  ```
  Inform: 2.64e-11 | Reminder: 1.50e-12 | Promotion: 4.79e-12
  Predicted class: Inform
  ```

### Notes
- You can tweak the `num_merges` variable or the Add-K constant to observe different vocabularies and smoothing behavior.
- All intermediate structures (counts and probabilities) are printed for inspection; feel free to extend logging or add plots if needed.

