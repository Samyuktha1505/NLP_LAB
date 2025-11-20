# ASSIGNMENT-1: Telugu Text Processing

This assignment focuses on basic text processing tasks for Telugu language data, including sentence tokenization, word tokenization, and corpus statistics. The tokenized sentences are saved in Parquet format with compression for efficient storage.

## Objectives
- Load and process Telugu text from IndicCorpV2 dataset
- Implement sentence tokenization for Telugu text
- Implement word/token tokenization with support for Telugu script, URLs, emails, dates, numbers, and punctuation
- Save tokenized sentences to Parquet format with compression
- Perform character-level analysis
- Calculate corpus statistics (TTR, average word length, average sentence length)

## Files
- `ASSIGN_1_1ST.ipynb` - Initial implementation notebook
- `NEW_ASSIGN_1_1ST.ipynb` - Updated implementation with improved tokenization
- `tokenized_sentences_parquet.ipynb` - **Recommended**: Implementation with Parquet output format
- `telugu_dataset.txt` - Processed Telugu corpus (text format)
- `telugu_tokenized_sentences.parquet` - Tokenized sentences in Parquet format (compressed)

## Key Features

### Data Processing Pipeline
1. **Load Dataset**: Stream Telugu text from IndicCorpV2 dataset
2. **Sentence Splitting**: Split paragraphs into sentences using regex pattern `(?<=[.!?])\s+`
3. **Word Tokenization**: Tokenize each sentence into words/tokens
4. **Format Output**: Join tokens with spaces to form tokenized sentences
5. **Save to Parquet**: Store in compressed Parquet format for efficient storage

### Sentence Tokenization
- Uses regex pattern `(?<=[.!?])\s+` to split sentences
- Handles Telugu punctuation marks

### Word Tokenization
Handles multiple text types:
- URLs (`https?://\S+`)
- Email addresses
- Dates (DD/MM/YYYY or DD-MM-YYYY)
- Decimal numbers and integers
- Telugu script (`[\u0C00-\u0C7F]+`)
- English words
- Punctuation marks

### Parquet Format
- **Format**: Each line contains one tokenized sentence (space-separated tokens)
- **Compression**: Snappy compression for fast read/write and good compression ratio
- **Benefits**: 
  - Efficient storage for large datasets
  - Fast reading and writing
  - Columnar storage format
  - Preserves data types

### Statistics Calculated
- Number of sentences
- Total words/tokens
- Unique tokens
- Number of characters
- Average word length
- Average sentence length
- Type-Token Ratio (TTR)

## Usage

### Using the Parquet Notebook (Recommended)
```python
# Run tokenized_sentences_parquet.ipynb
# This will:
# 1. Load the dataset
# 2. Split paragraphs into sentences
# 3. Tokenize each sentence
# 4. Save to telugu_tokenized_sentences.parquet
```

### Reading the Parquet File
```python
import pandas as pd

# Read tokenized sentences from parquet
df = pd.read_parquet('telugu_tokenized_sentences.parquet')

# Access tokenized sentences
for idx, row in df.iterrows():
    tokenized_sentence = row['tokenized_sentence']
    # Each sentence is space-separated tokens
    tokens = tokenized_sentence.split()
```

### Basic Processing
1. Load the dataset using `load_dataset("ai4bharat/IndicCorpV2", name="indiccorp_v2", split="tel_Telu", streaming=True)`
2. Process text through sentence and word tokenizers
3. Save tokenized sentences to Parquet format
4. Calculate and display corpus statistics

## Dependencies
- `datasets` - For loading IndicCorpV2 dataset
- `pandas` - For DataFrame operations and Parquet support
- `pyarrow` - For Parquet file format support
- `re` - For regex-based tokenization

## Output Format

### Parquet File Structure
- **Column**: `tokenized_sentence` (string)
- **Content**: Each row contains one tokenized sentence with space-separated tokens
- **Example**: `"అమెరికా అధ్యక్షుడు డొనాల్డ్ ట్రంప్ కు ."`

### File Naming
- Output file: `telugu_tokenized_sentences.parquet`
- Compression: Snappy (fast, efficient)

## Results
The processed dataset contains Telugu sentences with comprehensive tokenization supporting multiple text types and scripts. The tokenized sentences are stored efficiently in Parquet format, making it easy to load and process for downstream NLP tasks.

**Sample Output:**
- Total sentences: Variable (depends on dataset size)
- File format: Parquet with Snappy compression
- Each sentence: Space-separated tokenized words
