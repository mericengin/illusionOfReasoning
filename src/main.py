import pandas as pd
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

# Initialize the client so it can authenticate using your .env file
client = OpenAI()

def get_llm_response(text):
    prompt = f"You are a strictly logical evaluation system. Read the premises and determine if the conclusion is logically VALID or INVALID. Reply ONLY with the exact word 'VALID' or 'INVALID'.\n\nSyllogism:\n{text}\n\nAnswer:"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Use a fast, cost-effective model for the baseline
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,     # Enforce deterministic outputs
            max_tokens=2         # Restrict output length to prevent conversational bloat
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"API Error: {e}")
        return "ERROR"

def run_inference():
    df = pd.read_csv("data/data.csv")
    
    # Initialize output columns
    df['pred_standard_gpt-4o-mini'] = ""
    df['pred_scrambled_gpt-4o-mini'] = ""

    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        
        # Inference for standard text
        df.at[index, 'pred_standard_gpt-4o-mini'] = get_llm_response(row['syllogism'])
        time.sleep(0.5) # Rate limiting buffer
        
        # Inference for scrambled text
        df.at[index, 'pred_scrambled_gpt-4o-mini'] = get_llm_response(row['scrambled_syllogisms'])
        time.sleep(0.5)
        
    df.to_csv("data/results.csv", index=False)
    print("Inference complete. Results saved to data/results.csv")

run_inference()