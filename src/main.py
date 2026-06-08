import pandas as pd
from openai import OpenAI
import google.generativeai as genai
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- Client Initialization ---
# OpenAI automatically finds OPENAI_API_KEY in the environment
openai_client = OpenAI()

# Gemini requires explicit configuration and parameter structuring
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Set temperature to 0.0 for deterministic output
gemini_config = genai.types.GenerationConfig(
    temperature=0.0, 
    max_output_tokens=2
)
gemini_model = genai.GenerativeModel('gemini-1.5-flash', generation_config=gemini_config)

# --- Prompt Template ---
PROMPT_TEMPLATE = """You are a strictly logical evaluation system. Read the premises and determine if the conclusion is logically VALID or INVALID. Reply ONLY with the exact word 'VALID' or 'INVALID'.

Syllogism:
{text}

Answer:"""

# --- Inference Functions ---
def get_openai_response(text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "ERROR"

def get_gemini_response(text):
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip().upper()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "ERROR"

# --- Main Execution ---
def run_inference():
    df = pd.read_csv("data/data.csv")
    
    # Initialize output columns for both architectures
    df['pred_standard_gpt-4o-mini'] = ""
    df['pred_scrambled_gpt-4o-mini'] = ""
    df['pred_standard_gemini-1.5-flash'] = ""
    df['pred_scrambled_gemini-1.5-flash'] = ""

    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        
        # OpenAI Inference
        df.at[index, 'pred_standard_gpt-4o-mini'] = get_openai_response(row['syllogism'])
        time.sleep(0.5) 
        df.at[index, 'pred_scrambled_gpt-4o-mini'] = get_openai_response(row['scrambled_syllogisms'])
        time.sleep(0.5)
        
        # Gemini Inference
        df.at[index, 'pred_standard_gemini-1.5-flash'] = get_gemini_response(row['syllogism'])
        time.sleep(0.5) 
        df.at[index, 'pred_scrambled_gemini-1.5-flash'] = get_gemini_response(row['scrambled_syllogisms'])
        time.sleep(0.5)
        
    df.to_csv("data/results.csv", index=False)
    print("\nInference complete. Results saved to data/results.csv")

if __name__ == "__main__":
    run_inference()