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
```

## Installation & Setup

1. Clone the repository and set up your virtual environment:

pip install -r requirements.txt
python -m spacy download en_core_web_sm

2. **Configure Authentication:** Create a .env file in the root directory and add your API key (for each LLM you want to try out):

OPENAI_API_KEY="sk-your-api-key"
...

## Execution Pipeline

1. Data Generation 
Run the scrambling script to generate the parallel OOV dataset:

python scrambler.py

Compile the text arrays into a target DataFrame:

python processing.py

2. Inference

Execute the zero-shot prompting pipeline against the specified LLM endpoint (default: gpt-4o-mini). The script utilizes deterministic parameters (temperature=0.0).

python run_inference.py

**Note:** Once inference completes, you must manually add a ground_truth column to data/results.csv labeling each row as strictly VALID or INVALID before running the evaluation.

3. Evaluation

The evaluation script computes the model accuracy for both distributions. Given $N$ samples, accuracy is defined as the mean of the indicator function where the predicted label $\hat{y}_i$ matches the ground truth $y_i$:

$$Acc = \frac{1}{N}\sum_{i=1}^{N} \mathbb{I}(\hat{y}_i = y_i)$$

The primary evaluation metric is the performance delta ($\Delta$) between the standard and scrambled distributions:

$$\Delta = Acc_{standard} - Acc_{scrambled}$$

A statistically significant $\Delta > 0$ indicates a reliance on distributional semantics rather than robust content-neutral parsing.

To calculate and output the final metrics:

python evaluation.py