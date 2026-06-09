import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_visualizations():
    # 1. Load the results
    try:
        df = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        print("Error: Could not find data/results.csv. Run inference first.")
        return

    if 'ground_truth' not in df.columns:
        print("Error: 'ground_truth' column not found in results.csv.")
        print("Please manually add this column and label each row as 'VALID' or 'INVALID'.")
        return

    # 2. Calculate Accuracies
    models = {
        'GPT-4o-Mini': ('pred_standard_gpt-4o-mini', 'pred_scrambled_gpt-4o-mini'),
        'Gemini Flash Lite': ('pred_standard_gemini-flash-lite', 'pred_scrambled_gemini-flash-lite'),
        'GPT-5 Mini': ('pred_standard_gpt-5-mini', 'pred_scrambled_gpt-5-mini'),
    }

    results_data = []
    delta_data = []

    for model_name, (std_col, scr_col) in models.items():
        # Calculate percentage accuracy
        std_acc = (df[std_col] == df['ground_truth']).mean() * 100
        scr_acc = (df[scr_col] == df['ground_truth']).mean() * 100
        delta = std_acc - scr_acc
        
        results_data.append({'Model': model_name, 'Condition': 'Standard', 'Accuracy': std_acc})
        results_data.append({'Model': model_name, 'Condition': 'Scrambled', 'Accuracy': scr_acc})
        delta_data.append({'Model': model_name, 'Performance Drop (\u0394)': delta})

    results_df = pd.DataFrame(results_data)
    delta_df = pd.DataFrame(delta_data)

    # Set visualization style
    sns.set_theme(style="whitegrid")
    
    # 3. Create Figure 1: Accuracy Comparison (Grouped Bar Chart)
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(
        data=results_df, 
        x='Model', 
        y='Accuracy', 
        hue='Condition',
        palette=['#4C72B0', '#C44E52']
    )
    plt.title('LLM Deductive Reasoning: Standard vs. Scrambled Vocabulary', fontsize=14, pad=15)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    plt.ylim(0, 105) # Give a little headroom above 100%
    
    # Add data labels
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3)
        
    plt.tight_layout()
    plt.savefig('data/accuracy_comparison.png', dpi=300)
    plt.close()
    print("Saved accuracy comparison chart to data/accuracy_comparison.png")

    # 4. Create Figure 2: Performance Delta
    plt.figure(figsize=(8, 5))
    ax2 = sns.barplot(
        data=delta_df, 
        x='Model', 
        y='Performance Drop (\u0394)', 
        palette=['#55A868', '#DD8452'],
        hue='Model',
        legend=False
    )
    plt.title('Performance Drop on Scrambled Logic (\u0394)', fontsize=14, pad=15)
    plt.ylabel('Accuracy Drop (%)', fontsize=12)
    plt.xlabel('Model', fontsize=12)
    
    # Add a horizontal line at 0 for reference
    plt.axhline(0, color='black', linewidth=1)
    
    # Add data labels
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', padding=3)

    plt.tight_layout()
    plt.savefig('data/performance_delta.png', dpi=300)
    plt.close()
    print("Saved performance delta chart to data/performance_delta.png")

if __name__ == "__main__":
    create_visualizations()