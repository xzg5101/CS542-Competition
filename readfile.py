import pandas as pd
import numpy as np
import pickle
import os
with open('submission/predictions.pkl', 'rb') as f:
    preds = pickle.load(f)

print(preds)