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

# def 


