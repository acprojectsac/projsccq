import math, random, hashlib, os
import numpy as np
from typing import Iterable, List, Set, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from math import comb

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def read_data(filename):
    data_subset_list = []

    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  

            elems = [int(x) for x in line.split(",") if x != ""]
            subset_set = set(elems)
            data_subset_list.append(subset_set)

    return data_subset_list

def load_query(filename):
    query_thresholds_list = []
    query_subsets_list = []

    with open(PROJECT_PATH + filename, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  

            parts = line.split(maxsplit=1)
            t = int(parts[0])
            if len(parts) > 1 and parts[1].strip():
                subset_str = parts[1].strip()
                elems = [int(x) for x in subset_str.split(",") if x != ""]
                subset_set = set(elems)
            else:
                subset_set = set()

            query_thresholds_list.append(t)
            query_subsets_list.append(subset_set)

    return query_thresholds_list, query_subsets_list

def read_ground_truth_freq(ground_truth_freq_queries_path, num_queries):
    freqs = []
    with open(ground_truth_freq_queries_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            freqs.append(float(line))
    freqs = freqs[:num_queries]  
    assert len(freqs) == num_queries, f"Expected {num_queries} lines, but got {len(freqs)}"
    return freqs

def _normal_ccdf(z):
    try:
        import math
        return 0.5 * math.erfc(z / math.sqrt(2.0))
    except Exception:
        from math import erf, sqrt
        return 0.5 * (1.0 - erf(z / sqrt(2.0)))

def _gaussian_tail(mu: float, var: float, t: int) -> float:
    t = int(t)
    if t <= 0:
        return 1.0
    if var <= 1e-12:
        return 1.0 if mu >= (t - 0.5) else 0.0
    z = (t - 0.5 - mu) / math.sqrt(var)
    return float(_normal_ccdf(z))  
