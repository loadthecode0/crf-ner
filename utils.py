import numpy as np
import scipy as sp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import re, csv, string
from tqdm import tqdm
import pickle
import datetime

def start_cap(word: str) -> bool:
    return word[:1].isupper()

def end_ing(word: str) -> bool:
    return word[-3:] == 'ing'

def parse(filename:str, numlines=None):
    parsed_data = {}
    all_NER_tags = set()
    all_POS = set()
    i = 0

    with open(filename, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        
        for row in reader:
            i+=1
            if (i==numlines): break
            match = re.match(r'Sentence: (\d+)', row['Sentence #'])
            sentence_id = int(match.group(1)) if match else None
            words = row['Sentence'].split()
            pos_tags = eval(row['POS'])  # convert POS tags string to list
            ner_tags = eval(row['Tag'])  # convert NER tags string to list

            if (len(words) != len(pos_tags)) or (len(ner_tags) != len(pos_tags)) or ((len(words) != len(ner_tags))):
                print(len(words), len(pos_tags), len(ner_tags))
            else:
                if sentence_id not in parsed_data:
                    parsed_data[sentence_id] = {"Tokens": [], "POS": [], "NER_tags": []}
                    
                parsed_data[sentence_id]["Tokens"].extend(words)
                parsed_data[sentence_id]["POS"].extend(pos_tags)
                parsed_data[sentence_id]["NER_tags"].extend(ner_tags)

            # print(f'{len(ner_tags) == len(pos_tags)}')

            for t in ner_tags:
                all_NER_tags.add(t)

            for pos in pos_tags:
                all_POS.add(pos)

    all_NER_tags = list(all_NER_tags)
    all_POS = list(all_POS)
    parsed_data = pd.DataFrame(parsed_data).T

    return parsed_data, all_NER_tags, all_POS


