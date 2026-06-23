# Evaluation of LLM Deductive Reasoning: Distributional Semantics vs. Formal Logic

## 1. The Baseline Illusion (Standard Condition)
All three models evaluated (GPT-4o-Mini, Gemini Flash Lite, and GPT-5 Mini) achieved an identical **95.8% accuracy** on the standard syllogism dataset. 

In traditional NLP benchmarking, this high performance suggests that the models have successfully internalized deductive logic. However, because standard syllogisms utilize highly probable English words (e.g., "cats," "mammals," "animals"), the models can arrive at the correct answer through **distributional semantics**. They are predicting the next most statistically likely token based on training data co-occurrences, rather than actively traversing a formal logical tree.

## 2. The Structural Reality (Scrambled Condition)
To isolate true logical parsing from semantic pattern matching, real words were replaced with out-of-vocabulary (OOV) pseudo-words (e.g., "wugs," "florps"). This zeroed out the semantic weights, forcing the models to rely entirely on syntactic structure and logical operators (All, Some, No). 

The resulting performance drop ($\Delta$) exposes the degree to which each architecture relies on semantic priors:

* **GPT-4o-Mini ($\Delta$ = 25.0%):** The severe degradation to 70.8% accuracy strongly indicates that its reasoning is highly semantically dependent. Without familiar nouns and adjectives to anchor its predictions, its ability to parse formal logic collapses.

* **GPT-5 Mini ($\Delta$ = 12.5%):** This model halves the failure rate of its predecessor, dropping to 83.3% accuracy. Because this architecture utilizes latent reasoning (hidden Chain of Thought) prior to outputting an answer, its internal scratchpad allows it to maintain the relationships between abstract variables more effectively.

* **Gemini Flash Lite ($\Delta$ = 8.3%):** Gemini demonstrated the most robust syntactic parsing, maintaining an 87.5% accuracy. Its attention mechanisms appear less heavily weighted toward substantive tokens and better tuned to positional encodings and logical operators, allowing it to navigate abstract logic with minimal semantic reliance.

## 3. Conclusion
These results provide empirical evidence that while LLMs can effectively mimic logical reasoning when semantic priors are intact, stripping those priors reveals underlying structural vulnerabilities. Furthermore, different model architectures and inference techniques (such as latent Chain of Thought) exhibit drastically different levels of reliance on statistical pattern matching versus formal syntactic parsing.