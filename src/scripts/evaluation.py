import pandas as pd

def evaluate_results():
    # Load the completed inference results
    df = pd.read_csv("data/results.csv")
    
    # Calculate accuracy by comparing predictions to the ground truth
    # .mean() automatically converts True/False into a percentage decimal
    standard_acc = (df['pred_standard_gpt-4o-mini'] == df['ground_truth']).mean() * 100
    scrambled_acc = (df['pred_scrambled_gpt-4o-mini'] == df['ground_truth']).mean() * 100
    
    # Calculate the performance delta
    delta = standard_acc - scrambled_acc

    # Print the empirical findings
    print("\n=== EXPERIMENT RESULTS ===")
    print(f"Standard Dataset Accuracy : {standard_acc:.2f}%")
    print(f"Scrambled Dataset Accuracy: {scrambled_acc:.2f}%")
    print("==========================")
    
    if delta > 0:
        print(f"\nConclusion: The model's reasoning degraded by {delta:.2f}% when standard statistical patterns were removed.")
        print("This empirically supports Shanahan's argument that the LLM relies heavily on distributional semantics rather than pure content-neutral logic.")
    elif delta <= 0:
        print(f"\nConclusion: The model's reasoning did not degrade (Delta: {delta:.2f}%).")
        print("This challenges Shanahan's argument, suggesting the model may be successfully applying content-neutral formal logic.")

evaluate_results()