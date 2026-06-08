import pandas as pd
from openai import OpenAI
from google import genai
from google.genai import types
import time
import os
from dotenv import load_dotenv

load_dotenv()

# --- Client Initialization ---
openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

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
        response = gemini_client.models.generate_content(
            model='gemini-flash-lite-latest', 
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=2,
            )
        )
        return response.text.strip().upper()
    except Exception as e:
        print(f"Gemini API Error: {e}")
        return "ERROR"

# --- Main Execution ---
def run_inference():
    # 1. Read from the clean baseline dataset
    df = pd.read_csv("data/data.csv")
    
    # 2. Initialize all four output columns
    df['pred_standard_gpt-4o-mini'] = ""
    df['pred_scrambled_gpt-4o-mini'] = ""
    df['pred_standard_gemini-flash-lite'] = ""
    df['pred_scrambled_gemini-flash-lite'] = ""

    for index, row in df.iterrows():
        print(f"Processing row {index}...")
        
        # --- OpenAI Inference ---
        df.at[index, 'pred_standard_gpt-4o-mini'] = get_openai_response(row['syllogism'])
        time.sleep(0.5) 
        
        df.at[index, 'pred_scrambled_gpt-4o-mini'] = get_openai_response(row['scrambled_syllogisms'])
        time.sleep(0.5)
        
        # --- Gemini Inference ---
        df.at[index, 'pred_standard_gemini-flash-lite'] = get_gemini_response(row['syllogism'])
        time.sleep(4.1) # 4.1s buffer to stay under 15 requests/minute free tier limit
        
        df.at[index, 'pred_scrambled_gemini-flash-lite'] = get_gemini_response(row['scrambled_syllogisms'])
        time.sleep(4.1) # 4.1s buffer to stay under 15 requests/minute free tier limit
        
    # 3. Save everything to a single, consolidated results file
    df.to_csv("data/results.csv", index=False)
    print("\nInference complete. Both models' results saved to data/results.csv")

if __name__ == "__main__":
    run_inference()