import pandas as pd
from openai import OpenAI
import time
from dotenv import load_dotenv

load_dotenv()

# --- Client Initialization ---
openai_client = OpenAI()

# --- Configuration ---
# Update this string to the exact cheapest model name on your OpenAI dashboard
NEW_MODEL_NAME = "gpt-5-mini" 

PROMPT_TEMPLATE = """You are a strictly logical evaluation system. Read the premises and determine if the conclusion is logically VALID or INVALID. Reply ONLY with the exact word 'VALID' or 'INVALID'.

Syllogism:
{text}

Answer:"""

def get_openai_response(text, model_string):
    prompt = PROMPT_TEMPLATE.format(text=text)
    try:
        response = openai_client.chat.completions.create(
            model=model_string,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=2
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return "ERROR"

def append_new_model():
    # Safely load the EXISTING results file
    try:
        df = pd.read_csv("data/results.csv")
    except FileNotFoundError:
        print("Error: data/results.csv not found. Make sure you are in the root directory.")
        return

    # Define the new column names dynamically based on the model string
    col_standard = f"pred_standard_{NEW_MODEL_NAME}"
    col_scrambled = f"pred_scrambled_{NEW_MODEL_NAME}"

    # Initialize the new columns
    df[col_standard] = ""
    df[col_scrambled] = ""

    for index, row in df.iterrows():
        print(f"Processing row {index} for {NEW_MODEL_NAME}...")
        
        df.at[index, col_standard] = get_openai_response(row['syllogism'], NEW_MODEL_NAME)
        time.sleep(0.5) 
        
        df.at[index, col_scrambled] = get_openai_response(row['scrambled_syllogisms'], NEW_MODEL_NAME)
        time.sleep(0.5)
        
    # Save back to the same file, preserving all previous data
    df.to_csv("data/results.csv", index=False)
    print(f"\nSuccess! {NEW_MODEL_NAME} results appended to data/results.csv")

if __name__ == "__main__":
    append_new_model()