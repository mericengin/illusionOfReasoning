import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_full_visualizations():
    # Load both datasets
    df_zero = pd.read_csv("data/results.csv")
    df_cot = pd.read_csv("data/results_cot.csv")
    
    # 1. Prepare Data for Comparison
    results_data = []
    models = {
        'GPT-4o-Mini': ('pred_standard_gpt-4o-mini', 'pred_scrambled_gpt-4o-mini', 'cot_standard_gpt-4o-mini', 'cot_scrambled_gpt-4o-mini'),
        'Gemini Flash Lite': ('pred_standard_gemini-flash-lite', 'pred_scrambled_gemini-flash-lite', 'cot_standard_gemini-flash-lite', 'cot_scrambled_gemini-flash-lite'),
        'GPT-5 Mini': ('pred_standard_gpt-5-mini', 'pred_scrambled_gpt-5-mini', 'cot_standard_gpt-5-mini', 'cot_scrambled_gpt-5-mini')
    }

    for model, (z_std, z_scr, c_std, c_scr) in models.items():
        results_data.append({'Model': model, 'Condition': 'Zero-Shot Standard', 'Accuracy': (df_zero[z_std] == df_zero['ground_truth']).mean() * 100})
        results_data.append({'Model': model, 'Condition': 'Zero-Shot Scrambled', 'Accuracy': (df_zero[z_scr] == df_zero['ground_truth']).mean() * 100})
        results_data.append({'Model': model, 'Condition': 'CoT Standard', 'Accuracy': (df_cot[c_std] == df_cot['ground_truth']).mean() * 100})
        results_data.append({'Model': model, 'Condition': 'CoT Scrambled', 'Accuracy': (df_cot[c_scr] == df_cot['ground_truth']).mean() * 100})

    df_plot = pd.DataFrame(results_data)
    sns.set_theme(style="whitegrid")

    # Chart 1: Comparison Bar Plot
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_plot, x='Model', y='Accuracy', hue='Condition', palette='viridis')
    plt.title('Deductive Reasoning: Zero-Shot vs. Chain of Thought', fontsize=15)
    plt.ylim(0, 105)
    plt.savefig('data/cot_comparison_all.png', dpi=300)

    # Chart 2: CoT Gain Heatmap
    gain_data = []
    for model, (z_std, z_scr, c_std, c_scr) in models.items():
        gain_scr = ((df_cot[c_scr] == df_cot['ground_truth']).mean() - (df_zero[z_scr] == df_zero['ground_truth']).mean()) * 100
        gain_data.append({'Model': model, 'Gain (%)': gain_scr})
    
    gain_df = pd.DataFrame(gain_data)
    plt.figure(figsize=(8, 4))
    sns.barplot(data=gain_df, x='Model', y='Gain (%)', palette='coolwarm')
    plt.title('Performance Gain on Scrambled Logic via CoT', fontsize=14)
    plt.savefig('data/cot_gain_analysis.png', dpi=300)

    # Chart 3: Robustness Scatter Plot
    plt.figure(figsize=(8, 8))
    for model, (z_std, z_scr, c_std, c_scr) in models.items():
        std_acc = (df_cot[c_std] == df_cot['ground_truth']).mean() * 100
        scr_acc = (df_cot[c_scr] == df_cot['ground_truth']).mean() * 100
        plt.scatter(std_acc, scr_acc, label=model, s=100)
    
    plt.plot([60, 100], [60, 100], 'k--', label='Perfect Robustness')
    plt.xlabel('Standard Accuracy (%)')
    plt.ylabel('Scrambled Accuracy (%)')
    plt.title('Semantic Robustness Scatter (CoT)', fontsize=14)
    plt.legend()
    plt.savefig('data/robustness_scatter.png', dpi=300)

    print("All visualizations saved to /data folder!")

if __name__ == "__main__":
    create_full_visualizations()