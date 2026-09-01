import pandas as pd
from openai import OpenAI
from google import genai
import math
import time
import os
import re
from dotenv import load_dotenv

load_dotenv()

openai_client = OpenAI()
gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

FEW_SHOT_MESSAGES = [
    {"role": "system", "content": "You are a strictly logical evaluation system. Output only VALID or INVALID."},
    {"role": "user", "content": "Syllogism: All snarks are mimsy. All mimsy florps are zoptic. Therefore, some snarks are zoptic."},
    {"role": "assistant", "content": "INVALID"},
    {"role": "user", "content": "Syllogism: All wugs have glorps. All florps are wugs. Therefore, all florps have glorps."},
    {"role": "assistant", "content": "VALID"}
]

def extract_label(text):
    matches = re.findall(r'\b(VALID|INVALID)\b', text.upper())
    return matches[-1] if matches else "ERROR"

def get_openai_response(text, model_string, use_few_shot=False):
    if pd.isna(text) or text == "N/A": return "N/A", 0.0
        
    if use_few_shot:
        messages = FEW_SHOT_MESSAGES.copy()
        if model_string == 'gpt-5-mini':
            messages[0]["role"] = "user" 
        messages.append({"role": "user", "content": f"Syllogism: {text}"})
    else:
        if model_string == 'gpt-5-mini':
            messages = [{"role": "user", "content": f"Output only VALID or INVALID.\n\nSyllogism: {text}"}]
        else:
            messages = [{"role": "system", "content": "Output only VALID or INVALID."}, {"role": "user", "content": f"Syllogism: {text}"}]
    
    supports_logprobs = (model_string == 'gpt-4o-mini')
    
    while True:
        try:
            kwargs = {"model": model_string, "messages": messages}
            if supports_logprobs:
                kwargs["max_completion_tokens"] = 10
                kwargs["logprobs"] = True
                kwargs["top_logprobs"] = 1
                
            response = openai_client.chat.completions.create(**kwargs)
            ans = extract_label(response.choices[0].message.content)
            
            if supports_logprobs and response.choices[0].logprobs and ans != "ERROR":
                raw_logprob = response.choices[0].logprobs.content[0].logprob
                confidence = round(math.exp(raw_logprob) * 100, 1)
            else:
                confidence = "N/A"
                
            if ans in ["VALID", "INVALID"]:
                return ans, confidence
        except Exception as e:
            print(f"  [OpenAI Rate Limit] Waiting 3s... ({e})")
            time.sleep(3)

def get_gemini_response(text, use_few_shot=False):
    if pd.isna(text) or text == "N/A": return "N/A"
        
    if use_few_shot:
        prompt = "Examples:\nLogic: All snarks are mimsy... -> INVALID\nLogic: All wugs have glorps... -> VALID\n\nEvaluate this. Output only VALID or INVALID.\nSyllogism: " + text
    else:
        prompt = "Evaluate the following logic. Output only VALID or INVALID.\nSyllogism: " + text
        
    while True:
        try:
            response = gemini_client.models.generate_content(model='gemini-flash-lite-latest', contents=prompt)
            ans = extract_label(response.text)
            if ans in ["VALID", "INVALID"]:
                return ans
        except Exception as e:
            print(f"  [Gemini Rate Limit] Waiting 6s... ({e})")
            time.sleep(6)

def run_advanced_experiments():
    df = pd.read_csv("data/data.csv")
    out_df = df.copy()
    models = ['gpt-4o-mini', 'gpt-5-mini']
    
    print("Starting Advanced Inference...")
    for index, row in df.iterrows():
        print(f"Processing row {index} ({row['type']})...")
        
        if row['type'] == 'standard':
            for m in models:
                ans_de, conf_de = get_openai_response(row['syllogism_de'], m)
                ans_tr, conf_tr = get_openai_response(row['syllogism_tr'], m)
                out_df.at[index, f'de_{m}'], out_df.at[index, f'conf_de_{m}'] = ans_de, conf_de
                out_df.at[index, f'tr_{m}'], out_df.at[index, f'conf_tr_{m}'] = ans_tr, conf_tr
            
            out_df.at[index, 'de_gemini'] = get_gemini_response(row['syllogism_de'])
            time.sleep(4.5) 
            out_df.at[index, 'tr_gemini'] = get_gemini_response(row['syllogism_tr'])
            time.sleep(4.5)
            
            for m in models:
                ans, conf = get_openai_response(row['scrambled_en'], m, use_few_shot=True)
                out_df.at[index, f'fewshot_scrambled_{m}'] = ans
                out_df.at[index, f'conf_fs_scrambled_{m}'] = conf
                
            out_df.at[index, 'fewshot_scrambled_gemini'] = get_gemini_response(row['scrambled_en'], use_few_shot=True)
            time.sleep(4.5) 
            
        elif row['type'] == 'belief_bias':
            for m in models:
                ans, conf = get_openai_response(row['syllogism_en'], m)
                out_df.at[index, f'belief_bias_{m}'] = ans
                out_df.at[index, f'conf_bb_{m}'] = conf
                
            out_df.at[index, 'belief_bias_gemini'] = get_gemini_response(row['syllogism_en'])
            time.sleep(4.5)

    out_df.to_csv("data/results_advanced.csv", index=False)
    print("\nAdvanced evaluation complete! Saved to data/results_advanced.csv")

if __name__ == "__main__":
    run_advanced_experiments()