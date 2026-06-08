# LLM Reasoning Evaluation: Content-Neutral Logic vs. Distributional Semantics

This repository contains an empirical pipeline designed to evaluate the functional linguistic competence of Large Language Models (LLMs). Specifically, it tests whether models apply content-neutral formal logic to deductive reasoning tasks or rely on statistical pattern completion (distributional semantics). 

Theoretical grounding is drawn from the frameworks established by the "Talking About Large Language Models" paper by Shanahan, differentiating between formal syntactic mastery and actual cognitive reasoning.

## Methodology

The pipeline isolates formal logic from pre-training distributional priors by employing parallel datasets:
1.  **Standard Dataset:** Valid and invalid categorical/property syllogisms utilizing standard English vocabulary.
2.  **Scrambled Dataset:** A structurally identical dataset where substantive tokens (nouns, proper nouns, adjectives) are systematically mapped to out-of-vocabulary (OOV) pseudo-words (e.g., *florps*, *blickets*) using POS tagging. Logical operators and syntactic trees remain identical.

By zeroing out the statistical weight of the substantive tokens, the model is forced to evaluate the purely structural validity of the logic.

## Project Structure

```text
├── data/
│   ├── texts/
│   │   ├── syllogisms.txt              # Baseline dataset (Standard vocabulary)
│   │   └── syllogisms_scrambled.txt    # Generated dataset (Pseudo-word vocabulary)
│   └── data.csv                        # Merged inference inputs
├── docs/
│   └── LLMs_Perspectives f...pdf      # Core seminar literature proposal
├── src/
│   ├── notebooks/
│   │   ├── experiment.ipynb            # Interactive inference execution
│   │   └── visualizations.ipynb        # Data visualization prototyping
│   └── scripts/
│       ├── evaluation.py               # Calculates accuracy and performance delta
│       ├── processing.py               # Compiles text files into standardized CSVs
│       ├── scrambler.py                # POS targeting and OOV substitution script
│       ├── visualizations.py           # Generates final plots for the presentation
│       └── main.py                     # Primary pipeline execution
├── .env                                # Environment variables
├── .gitignore
├── README.MD
└── requirements.txt
