import pandas as pd
from openai import OpenAI
from google import genai
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()

# --- Client Initialization ---
openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# --- CoT Prompt Template ---
COT_PROMPT = """You are a strictly logical evaluation system. 
Let's think step by step. 
First, map the premises to variables (A, B, C). 
Second, determine if the conclusion necessarily follows. 
Finally, output the exact word 'VALID' or 'INVALID' as the very last word of your response.

Syllogism:
{text}"""

# --- Parsing Function ---
def extract_label(text):
    # Scans the model's full thought process and extracts the final answer
    matches = re.findall(r'\b(VALID|INVALID)\b', text.upper())
    if matches:
        return matches[-1] # Grabs the last occurrence
    return "ERROR"

# --- Inference Functions ---
def get_openai_response(text, model_string):
    prompt = COT_PROMPT.format(text=text)
    try:
        response = openai_client.chat.completions.create(
            model=model_string,
            messages=[{"role": "user", "content": prompt}]
            # Notice: No token limits here so the model can think!
        )
        return extract_label(response.choices[0].message.content)
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "ERROR"

def get_gemini_response(text):
    prompt = COT_PROMPT.format(text=text)
    try:
        response = gemini_client.models.generate_content(
            model='gemini-flash-lite-latest', 
            contents=prompt
            # Notice: No token limits here either!
        )
        return extract_label(response.text)
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "ERROR"

# --- Main Execution ---
def run_cot_inference():
    # We read from results.csv so we can carry over your ground_truth column!
    df = pd.read_csv("data/results.csv")
    
    # Create a fresh DataFrame just for CoT results
    out_df = pd.DataFrame({
        'syllogism': df['syllogism'],
        'scrambled_syllogisms': df['scrambled_syllogisms'],
        'ground_truth': df['ground_truth']
    })

    models = ['gpt-4o-mini', 'gpt-5-mini']

    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        
        # OpenAI Models
        for m in models:
            out_df.at[index, f'cot_standard_{m}'] = get_openai_response(row['syllogism'], m)
            time.sleep(0.5) 
            out_df.at[index, f'cot_scrambled_{m}'] = get_openai_response(row['scrambled_syllogisms'], m)
            time.sleep(0.5)
            
        # Gemini Model
        out_df.at[index, 'cot_standard_gemini-flash-lite'] = get_gemini_response(row['syllogism'])
        time.sleep(4.1) # Keeping the Free Tier safety buffer
        out_df.at[index, 'cot_scrambled_gemini-flash-lite'] = get_gemini_response(row['scrambled_syllogisms'])
        time.sleep(4.1)
        
    # Save to a NEW file so your baseline data is completely safe
    out_df.to_csv("data/results_cot.csv", index=False)
    print("\nCoT inference complete! Saved to data/results_cot.csv")

if __name__ == "__main__":
    run_cot_inference()