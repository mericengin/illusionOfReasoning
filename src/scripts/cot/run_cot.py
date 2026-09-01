import pandas as pd
from openai import OpenAI
from google import genai
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

COT_PROMPT = """You are a strictly logical evaluation system. 
Let's think step by step. 
First, map the premises to variables (A, B, C). 
Second, determine if the conclusion necessarily follows. 
Finally, output the exact word 'VALID' or 'INVALID' as the very last word of your response.

Syllogism:
{text}"""

def extract_label(text):
    if text == "N/A": return "N/A"
    matches = re.findall(r'\b(VALID|INVALID)\b', text.upper())
    return matches[-1] if matches else "ERROR"

def get_openai_response(text, model_string):
    if pd.isna(text) or text == "N/A": return "N/A"
    
    prompt = COT_PROMPT.format(text=text)
    
    while True:
        try:
            response = openai_client.chat.completions.create(
                model=model_string,
                messages=[{"role": "user", "content": prompt}]
            )
            ans = extract_label(response.choices[0].message.content)
            if ans in ["VALID", "INVALID"]:
                return ans
        except Exception as e:
            print(f"  [OpenAI Rate Limit] Waiting 3s... ({e})")
            time.sleep(3)

def get_gemini_response(text):
    if pd.isna(text) or text == "N/A": return "N/A"
    
    prompt = COT_PROMPT.format(text=text)
    
    while True:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-flash-lite-latest', 
                contents=prompt
            )
            ans = extract_label(response.text)
            if ans in ["VALID", "INVALID"]:
                return ans
        except Exception as e:
            print(f"  [Gemini Rate Limit] Waiting 6s... ({e})")
            time.sleep(6)

def run_cot_inference():
    df = pd.read_csv("data/results.csv")
    
    out_df = pd.DataFrame({
        'syllogism_en': df['syllogism_en'],
        'scrambled_en': df['scrambled_en'],
        'ground_truth': df['ground_truth']
    })

    models = ['gpt-4o-mini', 'gpt-5-mini']
    print("Starting Chain-of-Thought Inference...")
    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        
        # OpenAI Models
        for m in models:
            out_df.at[index, f'cot_standard_{m}'] = get_openai_response(row['syllogism_en'], m)
            out_df.at[index, f'cot_scrambled_{m}'] = get_openai_response(row['scrambled_en'], m)
            
        # Gemini Model
        out_df.at[index, 'cot_standard_gemini-flash-lite'] = get_gemini_response(row['syllogism_en'])
        time.sleep(4.5) 
        out_df.at[index, 'cot_scrambled_gemini-flash-lite'] = get_gemini_response(row['scrambled_en'])
        time.sleep(4.5)
        
    out_df.to_csv("data/results_cot.csv", index=False)
    print("\nCoT inference complete! Saved to data/results_cot.csv")

if __name__ == "__main__":
    run_cot_inference()