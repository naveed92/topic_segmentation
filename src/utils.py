from itertools import islice
import numpy as np

# Sliding window function
def window(seq, n=3):
    """https://stackoverflow.com/questions/6822725/rolling-or-sliding-window-iterator
    Returns a sliding window of width n over data from the iterable seq"""
    
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
# Compute depth scores
def get_depths(scores):
    """Given a sequence of coherence scores of length n, compute a sequence of depth scores of similar length"""
    
    def climb(seq, i, mode='left'):
        """Given a sequence seq of values and index i, advance the index either to the right or left while the 
        value keeps increasing, then return the value at new index
        """
        if mode == 'left':
            while True:
                curr = seq[i]
                if i == 0:
                    return curr
                i = i-1
                if not seq[i] > curr:
                    return curr

        if mode == 'right':
            while True:
                curr = seq[i]
                if i == (len(seq)-1):
                    return curr
                i = i+1
                if not seq[i] > curr:
                    return curr
    
    depths = []
    for i in range(len(scores)):
        score = scores[i]
        l_peak = climb(scores, i, mode='left')
        r_peak = climb(scores, i, mode='right')
        depth = 0.5 * (l_peak + r_peak - (2*score))
        depths.append(depth)
        
    return np.array(depths)


from scipy.signal import argrelmax

# Filter out local maxima
def get_local_maxima(depth_scores, order=1):
    """Given a sequence of depth scores, return a filtered sequence where only local maxima 
    selected based on the given order"""

    maxima_ids = argrelmax(depth_scores, order=order)[0]
    filtered_scores = np.zeros(len(depth_scores))
    filtered_scores[maxima_ids] = depth_scores[maxima_ids]
    return filtered_scores

# Automatic threshold computation
def compute_threshold(scores):
    """From Texttiling: https://aclanthology.org/J97-1003.pdf
    Automatically compute an appropriate threshold given a sequence of depth scores
    """
    
    s = scores[np.nonzero(scores)]
    threshold = np.mean(s) - (np.std(s) / 2)
    # threshold = np.mean(s) - (np.std(s))
    return threshold

def get_threshold_segments(scores, threshold=0.1):
    """Given a sequence of depth scores, return indexes where the value is greater than the threshold"""
    segment_ids = np.where(scores >= threshold)[0]
    return segment_ids