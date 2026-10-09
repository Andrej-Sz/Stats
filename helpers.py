import pandas as pd
import numpy as np


def orient_comps(comps, scores):
    comps = np.array(comps, copy=True, dtype=float)
    scores = np.array(scores, copy=True, dtype=float)
    for index in range(comps.shape[0]):
        pivot = int(np.argmax(np.abs(comps[index])))
        if comps[index,pivot] < 0:
            comps[index] *=-1
            scores[:index] *=-1
    return comps, scores