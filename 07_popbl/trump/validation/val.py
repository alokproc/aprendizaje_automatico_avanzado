import sys, pickle
import numpy as np
from tensorflow.keras.models import load_model

with open("../chars.txt", "rb") as fp:
    chars = pickle.load(fp)
    
n_vocab = len(chars)
int_to_char = dict((i, c) for i, c in enumerate(chars))
char_to_int = dict((c, i) for i, c in enumerate(chars))

model = load_model('../models/model03.h5')

