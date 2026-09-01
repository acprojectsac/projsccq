import numpy as np
import random
import math, os
import scipy.cluster as cluster
from treelib import Tree, Node
import queue
from collections import deque
from typing import Iterable, List, Set, Tuple, Dict, Optional

from MinHasher import _MinHasher, compute_user_sigs_process, compute_query_sigs_process
from frequency_oracle import Frequency_oracle
from utils import load_query, read_data, read_ground_truth_freq
from set_partitioning_tree import Set_partitioning_tree

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

random.seed(0)
np.random.seed(1)

def run_ours(
    epsilon_list = (0.1, 0.5, 1, 2, 3, 4, 5, 6, 7, 8),
    domain_size = None, 
    alpha = 0.2, 
    query_subset_size: int = 20,
    rng_seed: int = 1,\
    dataset_name = None
):
    rng = random.Random(rng_seed)
    global PROJECT_PATH
    try:
        PROJECT_PATH
    except NameError:
        PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    assert dataset_name != None
    thresholds, queries = load_query(PROJECT_PATH + f"/datasets/{dataset_name}/queries_{query_subset_size}.txt")    
    user_sets = read_data(PROJECT_PATH + f"/datasets/{dataset_name}/{dataset_name}.txt")

    gt_freqs_full = read_ground_truth_freq(
        PROJECT_PATH + f"/datasets/{dataset_name}/{dataset_name}_ground_truth_freq_queries_{query_subset_size}.txt",
        num_queries=len(queries)
    )

    # 随机抽取最多 999 条查询
    num_queries = len(queries)
    idxs = rng.sample(range(num_queries), min(999, num_queries))
    queries = [queries[i] for i in idxs]
    thresholds = [thresholds[i] for i in idxs]
    gt_freqs_full = [gt_freqs_full[i] for i in idxs]

    U = len(user_sets)
    Q = len(queries)
    item_set = set(list(range(domain_size)))
    results_mse: Dict[float, float] = {}
    results_mre: Dict[float, float] = {}

    for eps_total in epsilon_list:
        jacc_mse_sum = 0.0
        jacc_pairs = 0
        query_counts_est = [0] * Q
        query_counts_true = [0] * Q
        cnt_est, cnt_true = 0, 0
        max_depth_d = 8  
        set_partitioning_tree = Set_partitioning_tree(
            data=user_sets,
            fanout=2,   
            item_set=item_set,
            domain_size=domain_size,
            alpha=alpha,
            epsilon=eps_total,
            hash_func_maximum_num=512
        )
        print('start build tree')
        set_partitioning_tree._build_set_partitioning_tree_dynamic(max_depth_d=max_depth_d)
        print('finish build tree')
        
        U_report = U

        print("开始回答查询...")
        for qi in range(Q):
            S_q = queries[qi]
            t   = thresholds[qi]

            cnt_true = 0
            for S_i in user_sets:
                if len(S_q.intersection(S_i)) >= t:
                    cnt_true += 1

            ans = set_partitioning_tree.query(query_subset=S_q, query_threshold=t)

            if isinstance(ans, (float, np.floating)) and 0.0 - 1e-12 <= ans <= 1.0 + 1e-12:
                cnt_est = int(round(max(0.0, min(1.0, float(ans))) * U))
            else:
                try:
                    cnt_est = int(round(float(ans)))
                except Exception:
                    cnt_est = 0

            query_counts_true[qi] = cnt_true
            query_counts_est[qi]  = max(0, min(U_report, cnt_est))

            # if qi % 20 == 0:
            #     jm = (jacc_mse_sum / jacc_pairs) if jacc_pairs > 0 else 'Undefined.'
            #     print('freq_est:', query_counts_est[qi] / U_report, 'freq_true:', cnt_true / U) 
            # if qi % 100 == 0:
            #     print(qi + 1, "queries have completed.")

        if jacc_pairs > 0:
            jacc_mse = jacc_mse_sum / jacc_pairs
        else:
            jacc_mse = float('nan')

        mse_sum = 0.0
        mre_sum = 0.0
        for i in range(Q):
            true_freq = query_counts_true[i] / U
            est_freq  = min(max(0.0, query_counts_est[i] / U_report), 1.0)
            mse_sum  += (true_freq - est_freq) ** 2

            true_cnt    = query_counts_true[i]
            est_cnt_hat = est_freq * U
            if true_cnt == 0:
                continue
            mre_sum += min(100, abs(true_cnt - est_cnt_hat) / true_cnt)

        results_mse[eps_total] = mse_sum / Q
        results_mre[eps_total] = mre_sum / Q
        # print(f"eps={eps_total}  mse={results_mse[eps_total]:.6g}  mre={results_mre[eps_total]:.6g}") 

    return results_mse, results_mre

if __name__ == '__main__':
    avg_mse_dict, avg_mre_dict = run_ours(
    epsilon_list=(0.1,0.5,1,2,3,4,5,6,7,8),
    L=512,         
    domain_size = 5298,   # uba 
    # domain_size = 1656,     # pos
    # domain_size = 2603,   # online retail
    # domain_size = 1128,   # movielens
    alpha=0.2,  
    query_subset_size=50,
    CLIP_SIZE_MAX = None,
    dataset_name = "uba"
    )
    for eps in avg_mse_dict:
        print(f"ε={eps:.1f}    mse={avg_mse_dict[eps]:.6f}    mre={avg_mre_dict[eps]:.6f}")