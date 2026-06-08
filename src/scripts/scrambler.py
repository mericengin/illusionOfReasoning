import spacy
import random
import os

# Load the NLP model
nlp = spacy.load("en_core_web_sm")

# Lexicon of pseudo-words
pseudo_nouns = ["florp", "blicket", "wug", "zibble", "dax", "toma", "glorp", "snark"]
pseudo_adjectives = ["frumious", "mimsy", "slithy", "vorpal", "glarfy", "zoptic"]

def scramble_syllogism(text):
    doc = nlp(text)
    scrambled_tokens = []
    
    available_nouns = random.sample(pseudo_nouns, len(pseudo_nouns))
    available_adjs = random.sample(pseudo_adjectives, len(pseudo_adjectives))
    mapping = {}

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            if token.lemma_ not in mapping:
                mapping[token.lemma_] = available_nouns.pop()
            
            # Pluralization handling
            replacement = mapping[token.lemma_] + ("s" if token.tag_ == "NNS" else "")
            scrambled_tokens.append(replacement)
            
        elif token.pos_ == "ADJ":
            if token.lemma_ not in mapping:
                mapping[token.lemma_] = available_adjs.pop()
            scrambled_tokens.append(mapping[token.lemma_])
            
        else:
            scrambled_tokens.append(token.text)
            
        if token.whitespace_:
            scrambled_tokens[-1] += token.whitespace_

    return "".join(scrambled_tokens)

file_path = os.path.join("data/texts", "syllogisms.txt")

try:
    with open(file_path, "r", encoding="utf-8") as file:
        lines = file.readlines()

    print("--- Dataset Generation Output ---\n")
    with open("syllogisms_scrambled.txt", "w", encoding="utf-8") as new_file:
        for line in lines:
            # Clean up stray quotes, commas, or newlines from the raw text
            clean_line = line.strip().strip('",').strip() 
            if clean_line:
                # Save the scrambled text to a variable
                scrambled_line = scramble_syllogism(clean_line)
                
                print(f"Standard : {clean_line}")
                print(f"Scrambled: {scrambled_line}\n")
        
                # Write the SCRAMBLED text to the file, and add a newline
                new_file.write(scrambled_line + "\n")
 
except FileNotFoundError:
    print(f"Error: Could not locate '{file_path}'. Check your directory structure.")