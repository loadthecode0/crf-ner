import numpy as np
import scipy as sp
from scipy.optimize import minimize
import pandas as pd
import matplotlib.pyplot as plt
import re, csv, string
from tqdm import tqdm
import pickle
import datetime
from itertools import starmap
from collections import Counter

def start_cap(token: str, curr_pos=None, t=None, T=None, y=None, y_=None) -> bool:
    return token[:1].isupper()

def end_ing(token: str, curr_pos=None, t=None, T=None, y=None, y_=None) -> bool:
    return token[-3:] == 'ing'

def is_punct(token: str, curr_pos=None, t=None, T=None, y=None, y_=None) -> bool:
    return token in ['.', ',', '``', '`', ':', '$']

def is_digit(token: str, curr_pos=None, t=None, T=None, y=None, y_=None) -> bool:
    return token.isdigit()

def is_start(token: str, curr_pos=None, t=None, T=None, y=None, y_=None) -> bool:
    return t==0

def is_end(token: str, curr_pos=None, t=None, T=None, y=None, y_=None) -> bool:
    return t==T-1


obs_funcs = [
    start_cap,
    end_ing,
    is_punct,
    is_digit,
    is_start,
    is_end
]

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


import numpy as np
import matplotlib.pyplot as plt

def plot_weights(W, n, p, o, ner_list, pos_list, obs_funcs):
    # Ensure correct indexing of W
    n2 = n * n
    n_p = n*p
    matrix_part = np.reshape(W[:n2], (n, n))  # Reshape properly
    extra_rows = np.reshape(W[n2:n2 + 2 * n], (2, n))  # Reshape properly
    p_array = np.reshape(W[n2 + 2 * n: n2 + 2 * n + n_p], (p, n))  # Ensure it’s 2D
    o_list = np.reshape(W[n2 + 2 * n + n_p:], (1, o))  # Ensure it’s 2D

    # Create a unified color scale for all parts
    all_values = np.concatenate([
        matrix_part.flatten(),
        extra_rows.flatten(),
        p_array.flatten(),
        o_list.flatten()
    ])
    vmin, vmax = np.min(all_values), np.max(all_values)

    # Create a new figure for the visualization
    fig, axs = plt.subplots(4, 1, figsize=(10, 7), gridspec_kw={'height_ratios': [n, 2, p, 1]})
    #Define common tick label styling
    tick_label_fontsize = 6  # Smaller font
    tick_label_rotation = 0  # Slanted labels
    # Plot the n x n matrix as a heatmap
    ax = axs[0]
    im1 = ax.imshow(matrix_part, cmap="viridis", aspect="auto")
    ax.set_title("Transition matrix", fontsize=tick_label_fontsize)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(ner_list, fontsize=tick_label_fontsize, rotation=tick_label_rotation)
    ax.set_yticklabels(ner_list, fontsize=tick_label_fontsize)
    fig.colorbar(im1, ax=ax, orientation="vertical")

    # Plot the extra rows as a heatmap
    ax = axs[1]
    im2 = ax.imshow(extra_rows, cmap="viridis", aspect="auto")
    ax.set_title("Special transitions", fontsize=tick_label_fontsize)
    ax.set_xticks(range(n))
    ax.set_yticks(range(2))
    ax.set_xticklabels(ner_list, fontsize=tick_label_fontsize, rotation=tick_label_rotation)
    ax.set_yticklabels(["From BOS", "To EOS"], fontsize=tick_label_fontsize)
    fig.colorbar(im2, ax=ax, orientation="vertical")

    # Plot the list p as a row matrix heatmap
    ax = axs[2]
    im3 = ax.imshow(p_array, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("POS emission", fontsize=tick_label_fontsize)
    ax.set_yticks(range(p))
    ax.set_xticks(range(n))
    ax.set_xticklabels(ner_list, fontsize=tick_label_fontsize, rotation=tick_label_rotation)
    ax.set_yticklabels(pos_list, fontsize=tick_label_fontsize)
    fig.colorbar(im3, ax=ax, orientation="vertical")

    # Plot the list of size o as a row matrix heatmap
    ax = axs[3]
    im4 = ax.imshow(o_list, cmap="viridis", aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_title("Observation functions", fontsize=tick_label_fontsize)
    ax.set_xticks(range(o))
    # ax.set_yticks([0])
    ax.set_xticklabels(obs_funcs)
    fig.colorbar(im4, ax=ax, orientation="vertical")

    # Show the updated visualization
    plt.tight_layout()
    plt.show()



def compute_class_weights(Y_train, smoothing_factor=1.0):
    """Computes smoothed class weights and normalizes so that the highest weight is 1."""
    label_counts = Counter(label for sentence in Y_train for label in sentence)
    total_labels = sum(label_counts.values())

    # Compute raw class weights
    raw_weights = {
        label: np.log(1 + smoothing_factor * (total_labels / count))
        for label, count in label_counts.items()
    }

    # Normalize so that the largest weight is 1
    alpha = 1.00
    max_weight = max(raw_weights.values())
    class_weights = {label: alpha*weight / max_weight for label, weight in raw_weights.items()}

    return class_weights