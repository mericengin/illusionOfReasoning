import pandas as pd
import numpy as np

def build_dataframe():
    syllogisms = []
    scrambled_syllogisms = []
    with open("data/texts/syllogisms.txt", "r") as f:
        for line in f.readlines():
            clean_line = line.strip()
            if clean_line:
                syllogisms.append(clean_line)

    with open("data/texts/syllogisms_scrambled.txt", "r") as f:
        for line in f.readlines():
            clean_line = line.strip()
            if clean_line:
                scrambled_syllogisms.append(clean_line)

    df = pd.DataFrame({'syllogism': syllogisms, 'scrambled_syllogisms': scrambled_syllogisms})
    df.to_csv("data/data.csv", index=False)
    

build_dataframe()