# ASSIGNMENT-2: Finite State Automata for English Noun Analysis

This assignment implements Deterministic Finite Automata (DFA) and Finite State Transducers (FST) to analyze English nouns, identifying singular and plural forms.

## Objectives
- Build a DFA to recognize valid English words (starting with uppercase, followed by lowercase)
- Implement an FST to analyze noun forms (singular/plural)
- Visualize automata using graphviz
- Analyze Brown corpus nouns for morphological patterns

## Files
- `ASSIGN_2_1ST.ipynb` - DFA implementation for word recognition
- `ASSIGN_2_2ND.ipynb` - FST implementation for noun analysis
- `brown_nouns.txt` - Brown corpus noun data
- `fst.png` - FST visualization
- `fst_dfa.png.png` - DFA visualization
- `noun_analysis_output.txt` - Analysis results

## Key Features

### DFA (Deterministic Finite Automaton)
- **States**: q0 (initial), q1, q2 (final), qReject
- **Transitions**:
  - q0 → q1: First character (uppercase or lowercase)
  - q1 → q2: Second character (lowercase only)
  - q2 → q2: Subsequent characters (lowercase only)
- **Purpose**: Validates word format (capitalized or all lowercase)

### FST (Finite State Transducer)
- **States**: q0, q1, q2, q3 (final), dead
- **Input/Output Symbols**: letter, s, e, y
- **Function**: Analyzes noun forms and outputs morphological tags
- **Patterns Handled**:
  - Words ending in `-es` (after s, x, z, ch, sh) → `+N+PL`
  - Words ending in `-ies` → `+N+PL` (root + y)
  - Words ending in `-s` → `+N+PL` (root)
  - Singular nouns → `+N+SG`

## Usage
1. Load Brown corpus nouns from `brown_nouns.txt`
2. Build and visualize DFA using `visual-automata` library
3. Implement FST for morphological analysis
4. Analyze nouns and generate output with tags

## Dependencies
- `visual-automata` - For automata visualization
- `automata-lib` - For automata construction
- `graphviz` - For diagram generation

## Results
The implementation successfully identifies singular and plural forms of English nouns from the Brown corpus, with visual representations of the automata structures.
