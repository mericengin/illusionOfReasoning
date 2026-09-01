import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_full_visualizations():
    # Load strictly the 24 standard items
    try:
        df_zero = pd.read_csv("data/results.csv").head(24)
        df_cot = pd.read_csv("data/results_cot.csv").head(24)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return
    
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

    # Chart 1: Comparison Bar Plot with clear labels
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(data=df_plot, x='Model', y='Accuracy', hue='Condition', palette='viridis')
    plt.title('Deductive Reasoning: Zero-Shot vs. Chain of Thought', fontsize=15, pad=15)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3, fontsize=9)
    plt.tight_layout()
    plt.savefig('data/cot_comparison_all.png', dpi=300)
    plt.close()
    print("Saved: data/cot_comparison_all.png")

    # Chart 2: CoT Gain Analysis
    gain_data = []
    for model, (z_std, z_scr, c_std, c_scr) in models.items():
        gain_scr = ((df_cot[c_scr] == df_cot['ground_truth']).mean() - (df_zero[z_scr] == df_zero['ground_truth']).mean()) * 100
        gain_data.append({'Model': model, 'Gain (%)': gain_scr})
    
    gain_df = pd.DataFrame(gain_data)
    plt.figure(figsize=(8, 4))
    ax2 = sns.barplot(data=gain_df, x='Model', y='Gain (%)', palette=['#4C72B0', '#C44E52', '#55A868'], edgecolor='black', linewidth=1, hue='Model', legend=False)
    plt.title('Performance Gain on Scrambled Logic via CoT', fontsize=14, pad=15)
    plt.ylabel('Gain (%)')
    plt.axhline(0, color='black', linewidth=1)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%+.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig('data/cot_gain_analysis.png', dpi=300)
    plt.close()
    print("Saved: data/cot_gain_analysis.png")

    # Chart 3: Robustness Scatter Plot
    plt.figure(figsize=(8, 8))
    visual_offsets = {'GPT-4o-Mini': 0, 'Gemini Flash Lite': -0.3, 'GPT-5 Mini': 0.3}
    markers = {'GPT-4o-Mini': 'o', 'Gemini Flash Lite': 's', 'GPT-5 Mini': '^'}

    for model, (z_std, z_scr, c_std, c_scr) in models.items():
        std_acc = (df_cot[c_std] == df_cot['ground_truth']).mean() * 100
        scr_acc = (df_cot[c_scr] == df_cot['ground_truth']).mean() * 100
        plt.scatter(std_acc + visual_offsets[model], scr_acc, label=model, s=150, marker=markers[model], alpha=0.8, edgecolor='black')
    
    plt.plot([60, 100], [60, 100], 'k--', label='Perfect Robustness')
    plt.xlabel('Standard Accuracy (%)')
    plt.ylabel('Scrambled Accuracy (%)')
    plt.title('Semantic Robustness Scatter (CoT)', fontsize=14, pad=15)
    plt.legend()
    plt.tight_layout()
    plt.savefig('data/robustness_scatter.png', dpi=300)
    plt.close()
    print("Saved: data/robustness_scatter.png")

if __name__ == "__main__":
    create_full_visualizations()