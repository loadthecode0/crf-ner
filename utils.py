import numpy as np
import scipy as sp
import re, csv, string

def start_cap(word: str) -> bool:
    return word[:1].isupper()

def end_ing(word: str) -> bool:
    return word[-3:] == 'ing'

# def 


