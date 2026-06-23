import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def create_cot_visualizations():
    df = pd.read_csv("data/results_cot.csv")

    models = {
        'GPT-4o-Mini': ('cot_standard_gpt-4o-mini', 'cot_scrambled_gpt-4o-mini'),
        'Gemini Flash Lite': ('cot_standard_gemini-flash-lite', 'cot_scrambled_gemini-flash-lite'),
        'GPT-5 Mini': ('cot_standard_gpt-5-mini', 'cot_scrambled_gpt-5-mini')
    }

    results_data = []
    
    for model_name, (std_col, scr_col) in models.items():
        std_acc = (df[std_col] == df['ground_truth']).mean() * 100
        scr_acc = (df[scr_col] == df['ground_truth']).mean() * 100
        
        results_data.append({'Model': model_name, 'Condition': 'Standard (CoT)', 'Accuracy': std_acc})
        results_data.append({'Model': model_name, 'Condition': 'Scrambled (CoT)', 'Accuracy': scr_acc})

    results_df = pd.DataFrame(results_data)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=results_df, 
        x='Model', 
        y='Accuracy', 
        hue='Condition',
        palette=['#7B4173', '#A55194'] # Different colors to distinguish from baseline
    )
    plt.title('Chain of Thought Reasoning: Standard vs. Scrambled Vocabulary', fontsize=14, pad=15)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.ylim(0, 105)
    
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f%%', padding=3)
        
    plt.tight_layout()
    plt.savefig('data/cot_accuracy.png', dpi=300)
    print("Saved CoT comparison chart to data/cot_accuracy.png")

if __name__ == "__main__":
    create_cot_visualizations()