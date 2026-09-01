import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_advanced_visualizations():
    try:
        df_adv = pd.read_csv("data/results_advanced.csv")
        # FIXED: Strictly take the 24 standard rows for baseline English comparison
        df_base = pd.read_csv("data/results.csv").head(24)
    except FileNotFoundError as e:
        print(f"Error loading CSV files: {e}")
        return

    models = {
        'GPT-4o-Mini': {
            'base_en': 'pred_standard_gpt-4o-mini',
            'adv_de': 'de_gpt-4o-mini',
            'adv_tr': 'tr_gpt-4o-mini',
            'adv_bb': 'belief_bias_gpt-4o-mini'
        },
        'Gemini Flash Lite': {
            'base_en': 'pred_standard_gemini-flash-lite',
            'adv_de': 'de_gemini',
            'adv_tr': 'tr_gemini',
            'adv_bb': 'belief_bias_gemini'
        },
        'GPT-5 Mini': {
            'base_en': 'pred_standard_gpt-5-mini',
            'adv_de': 'de_gpt-5-mini',
            'adv_tr': 'tr_gpt-5-mini',
            'adv_bb': 'belief_bias_gpt-5-mini'
        }
    }

    # --- 1. Data Prep: Cross-Lingual ---
    df_adv_std = df_adv[df_adv['type'] == 'standard'].head(24)
    lang_data = []

    for model_name, cols in models.items():
        acc_en = (df_base[cols['base_en']] == df_base['ground_truth']).mean() * 100
        acc_de = (df_adv_std[cols['adv_de']] == df_adv_std['ground_truth']).mean() * 100
        acc_tr = (df_adv_std[cols['adv_tr']] == df_adv_std['ground_truth']).mean() * 100

        lang_data.extend([
            {'Model': model_name, 'Language': 'English (Base)', 'Accuracy': acc_en},
            {'Model': model_name, 'Language': 'German', 'Accuracy': acc_de},
            {'Model': model_name, 'Language': 'Turkish', 'Accuracy': acc_tr}
        ])

    # --- 2. Data Prep: Belief Bias ---
    df_bb = df_adv[df_adv['type'] == 'belief_bias']
    bb_data = []

    for model_name, cols in models.items():
        acc_en = (df_base[cols['base_en']] == df_base['ground_truth']).mean() * 100
        acc_bb = (df_bb[cols['adv_bb']] == df_bb['ground_truth']).mean() * 100

        bb_data.extend([
            {'Model': model_name, 'Condition': 'Standard Logic', 'Accuracy': acc_en},
            {'Model': model_name, 'Condition': 'Belief Bias', 'Accuracy': acc_bb}
        ])

    sns.set_theme(style="whitegrid")

    # Chart 1: Cross-Lingual Grouped Bar
    plt.figure(figsize=(10, 6))
    ax1 = sns.barplot(data=pd.DataFrame(lang_data), x='Model', y='Accuracy', hue='Language', palette='viridis')
    plt.title('Cross-Lingual Deductive Reasoning Accuracy', fontsize=14, pad=15)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for container in ax1.containers:
        ax1.bar_label(container, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig('data/cross_lingual_accuracy.png', dpi=300)
    plt.close()
    print("Saved: data/cross_lingual_accuracy.png")

    # Chart 2: Belief Bias Grouped Bar
    plt.figure(figsize=(9, 6))
    ax2 = sns.barplot(data=pd.DataFrame(bb_data), x='Model', y='Accuracy', hue='Condition', palette=['#4C72B0', '#C44E52'])
    plt.title('Belief Bias Vulnerability: Logic vs. Real-World Facts', fontsize=14, pad=15)
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for container in ax2.containers:
        ax2.bar_label(container, fmt='%.1f%%', padding=3)
    plt.tight_layout()
    plt.savefig('data/belief_bias_vulnerability.png', dpi=300)
    plt.close()
    print("Saved: data/belief_bias_vulnerability.png")

if __name__ == "__main__":
    create_advanced_visualizations()