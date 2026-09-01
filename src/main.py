import pandas as pd
from openai import OpenAI
from google import genai
from google.genai import types
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def extract_label(text):
    matches = re.findall(r'\b(VALID|INVALID)\b', text.upper())
    return matches[-1] if matches else "ERROR"

def get_openai_response(text, model_string):
    if pd.isna(text) or text == "N/A": return "N/A"
    
    prompt = f"You are a strictly logical evaluation system. Read the premises and determine if the conclusion is logically VALID or INVALID. Reply ONLY with the exact word 'VALID' or 'INVALID'.\n\nSyllogism:\n{text}\n\nAnswer:"
    messages = [{"role": "user", "content": prompt}]
    
    while True:
        try:
            kwargs = {"model": model_string, "messages": messages}
            if model_string != 'gpt-5-mini': # GPT-5 Mini rejects temperature/max_tokens
                kwargs["temperature"] = 0.0
                kwargs["max_completion_tokens"] = 10
                
            response = openai_client.chat.completions.create(**kwargs)
            ans = extract_label(response.choices[0].message.content)
            if ans in ["VALID", "INVALID"]:
                return ans
        except Exception as e:
            print(f"  [OpenAI Rate Limit] Waiting 3s... ({e})")
            time.sleep(3)

def get_gemini_response(text):
    if pd.isna(text) or text == "N/A": return "N/A"
    
    prompt = f"You are a strictly logical evaluation system. Read the premises and determine if the conclusion is logically VALID or INVALID. Reply ONLY with the exact word 'VALID' or 'INVALID'.\n\nSyllogism:\n{text}\n\nAnswer:"
    
    while True:
        try:
            response = gemini_client.models.generate_content(
                model='gemini-flash-lite-latest', 
                contents=prompt,
                config=types.GenerateContentConfig(temperature=0.0, max_output_tokens=10)
            )
            ans = extract_label(response.text)
            if ans in ["VALID", "INVALID"]:
                return ans
        except Exception as e:
            print(f"  [Gemini Rate Limit] Waiting 6s... ({e})")
            time.sleep(6)

def run_inference():
    df = pd.read_csv("data/data.csv")
    models = ['gpt-4o-mini', 'gpt-5-mini']
    
    for m in models:
        df[f'pred_standard_{m}'] = ""
        df[f'pred_scrambled_{m}'] = ""
    df['pred_standard_gemini-flash-lite'] = ""
    df['pred_scrambled_gemini-flash-lite'] = ""

    print("Starting Baseline Inference...")
    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        
        # OpenAI Models
        for m in models:
            df.at[index, f'pred_standard_{m}'] = get_openai_response(row['syllogism_en'], m)
            df.at[index, f'pred_scrambled_{m}'] = get_openai_response(row['scrambled_en'], m)
        
        # Gemini Model
        df.at[index, 'pred_standard_gemini-flash-lite'] = get_gemini_response(row['syllogism_en'])
        time.sleep(4.5) 
        df.at[index, 'pred_scrambled_gemini-flash-lite'] = get_gemini_response(row['scrambled_en'])
        time.sleep(4.5)
        
    df.to_csv("data/results.csv", index=False)
    print("\nBaseline inference complete! Saved to data/results.csv")

if __name__ == "__main__":
    run_inference()