import time
import numpy as np
import random
import math, os
import scipy.cluster as cluster
from treelib import Tree, Node
import queue
from typing import Iterable, List, Set, Tuple, Dict, Optional
from collections import deque, defaultdict


from MinHasher import _MinHasher, compute_user_sigs_process, compute_query_sigs_process
from frequency_oracle import Frequency_oracle
from utils import _normal_ccdf, _gaussian_tail

random.seed(0)
np.random.seed(1)

class Node_partitioning(object):
    def __init__(self, this_node_item_set, fanout, domain_size=None, frequency = None, error = None):
        self.this_node_item_set = this_node_item_set
        self.fanout = fanout
        
        self.perturbed_reports = []
        
        assert len(this_node_item_set)//fanout >= 1, "the number of items in this node should be larger than fanout"
        self.item_num_groups = list(range(1, len(this_node_item_set), len(this_node_item_set)//fanout)) 
        self.item_num_groups_frequencies = np.zeros(len(self.item_num_groups), dtype=np.float64)
        self.left_user_ratio = 0
        self.error_coef = None

    def set_allocated_users(self, total_num, left_num, proportion):
        self.user_num = round(left_num * proportion)
        self.left_user_num = left_num - self.user_num
        index_l = total_num - left_num
        index_r =index_l + self.user_num - 1
        self.allocated_users = [index_l, index_r]

    def _set_node_queries(self, overall_item_frequency_list):
        pairs = sorted(((item, overall_item_frequency_list[item]) for item in self.this_node_item_set), \
            key=lambda x: (-x[1], x[0]))
        self.this_node_item_frequency_list = []
        self.this_node_item_to_idx_dict = {}
        self.this_node_idx_to_item_dict = {}
        for idx, (item, freq) in enumerate(pairs):
            self.this_node_item_frequency_list.append(freq)
            self.this_node_item_to_idx_dict[item] = idx
            self.this_node_idx_to_item_dict[idx] = item
        self.this_node_queries = []
        self.this_node_query_to_query_idx = {}
        current_subset = set()
        for item_idx, item_freq in enumerate(self.this_node_item_frequency_list):
            current_subset.add(self.this_node_idx_to_item_dict[item_idx])
            sub = frozenset(current_subset)
            self.this_node_queries.append(list(current_subset))
            self.this_node_query_to_query_idx[sub] = item_idx
    
    def _build_node_local_domain(self):
        local_items = sorted(list(self.this_node_item_set))
        g2l = {g: i for i, g in enumerate(local_items)}
        other_id = len(local_items)

        self._local_items = local_items         
        self._g2l = g2l
        self._other_id = other_id
        self._local_D = other_id + 1                    
    
    def _to_local_set(self, global_set, g2l, other_id):
        out = set()
        for x in global_set:
            out.add(g2l.get(x, other_id))
        return out
    
    def _pair_id_local(self, a_l: int, b_l: int, K: int) -> int:
        offset_a = a_l * (K - 1) - (a_l * (a_l + 1)) // 2
        return 1 + offset_a + (b_l - a_l - 1)
    
    def estimate_local_pairwise_freq(self, user_sets, users_range, epsilon):
        K = len(self._local_items)
        M = 1 + (K * (K - 1)) // 2

        l, r = users_range[0], users_range[1]
        sampled_ids = []

        for S in user_sets[l:r+1]:
            inter = [self._g2l[x] for x in S if x in self._g2l]
            if len(inter) >= 2:
                a_l, b_l = random.sample(inter, 2)
                if a_l > b_l:
                    a_l, b_l = b_l, a_l
                cat = self._pair_id_local(a_l, b_l, K)
            else:
                cat = 0
            sampled_ids.append(cat)

        fo = Frequency_oracle(
            data=sampled_ids,
            frequency_oracle_name='OUE',
            epsilon=epsilon,
            domain_size=M,
            merged_domain=None
        )
        freq_vec = fo.get_aggregated_frequency(smooth=True)  

        pair_freq = {(-1, -1): float(freq_vec[0])}

        idx = 1
        for a_l in range(K):
            for b_l in range(a_l + 1, K):
                freq = float(freq_vec[idx])
                a_g = self._local_items[a_l]
                b_g = self._local_items[b_l]
                pair_freq[(a_g, b_g)] = freq
                idx += 1

        self.local_pairwise_freq = pair_freq
        return pair_freq

def _report_oue_value(eps, d, value):
    p = 1 / 2
    q = 1 / (math.exp(eps) + 1)
    
    perturbed_vector = np.zeros(d)
    perturbed_vector[value] = 1
    random_matrix = np.random.uniform(0, 1, d)
    flip_to_1 = (perturbed_vector == 0) & (random_matrix < q)
    flip_to_0 = (perturbed_vector == 1) & (random_matrix < p)

    perturbed_vector[flip_to_1] = 1
    perturbed_vector[flip_to_0] = 0
        
    return perturbed_vector

class Set_partitioning_tree(object):
    def __init__(self, data, fanout, item_set, domain_size, alpha, epsilon, \
        hash_func_maximum_num = 512, use_hybrid_pxk_in_partial=True, use_dual_full=True):
        self.data = data
        self.fanout = fanout
        self.item_set = item_set
        self.epsilon = epsilon
        self.domain_size = domain_size
        self.user_num = len(data)
        self.alpha = alpha
        self.hash_func_maximum_num = hash_func_maximum_num
        self.hash_func_sample_number = self._determine_hash_func_sample_number()
        self.per_hash_func_epsilon = self.epsilon / self.hash_func_sample_number
        
        self.use_dual_full = use_dual_full          
        self.dual_split_ratio = 0.5         
        self.eps_oue_intersection = self.epsilon  
        self.use_hybrid_pxk_in_partial = use_hybrid_pxk_in_partial
        self.hybrid_pxk_weight = 0.5
        self.alpha_eta = 1   
    
    def _determine_hash_func_sample_number(self):

        return math.ceil(self.epsilon/2.1773189849)

    def _compute_pair_lifts_in_node(self, node, eps: float = 1e-12):

        items = sorted(list(node.data.this_node_item_set))

        MAX_K_FOR_PAIRLIFT = 1000   
        if len(items) > MAX_K_FOR_PAIRLIFT:
            return []  

        if len(items) < 2:
            return []

        U = self.underlying_users[1] - self.underlying_users[0] + 1
        U = max(1, U)

        p = {i: max(0.0, min(1.0, float(self.noisy_hist[i]) / U)) for i in items}

        lifts = []
        for ai in range(len(items)):
            a = items[ai]
            for bi in range(ai + 1, len(items)):
                b = items[bi]
                pab = node.data.local_pairwise_freq.get((a, b), None)
                if pab is None:
                    pab = node.data.local_pairwise_freq.get((b, a), 0.0)
                pab = max(0.0, float(pab))

                denom = max(eps, p[a] * p[b])
                lift = pab / denom
                lifts.append(((a, b), lift, pab, p[a], p[b]))

        lifts.sort(key=lambda x: (x[1], x[0][0], x[0][1]))
        return lifts

    def _items_order_from_lifts(self, node, lifts):
        score = defaultdict(float)
        m = len(lifts)

        for rank, (pair, lift, pab, pa, pb) in enumerate(lifts):
            w = float(m - rank)   
            a, b = pair
            score[a] += w
            score[b] += w

        ordered_items = sorted(
            list(node.data.this_node_item_set),
            key=lambda x: (-score.get(x, 0.0), x)  
        )
        return ordered_items, score

    def _mean_jaccard_from_sig(self, node, sig_q_ids):
        e = math.exp(self.per_hash_func_epsilon)
        p = 0.5
        q = 1.0 / (e + 1.0)  

        total_J = 0.0
        cnt = 0

        for usr_rep in node.data.perturbed_reports:
            matches = 0
            trials = 0
            for (s_idx, perturbed_rep) in usr_rep:
                y = sig_q_ids[s_idx]
                trials += 1
                if int(perturbed_rep[y]) == 1:
                    matches += 1
            if trials == 0:
                continue
            Zbar = matches / trials
            J_hat = (Zbar - q) / (p - q)
            J_hat = min(1.0, max(0.0, J_hat))
            total_J += J_hat
            cnt += 1

        if cnt == 0:
            return 0.0
        return total_J / float(cnt)

    def _propose_binary_split_by_pmi_serialization_and_MI_cut(self, node, min_child_size: int = 2, \
        eps: float = 1e-12, if_random_for_ablation_study=False):
        if if_random_for_ablation_study:
            return self._propose_binary_split_random(node=node,
                min_child_size=min_child_size)

        K = len(node.data.this_node_item_set)
        if K < 2 * min_child_size:
            return None

        if not hasattr(node.data, "_local_D"):
            node.data._build_node_local_domain()


        pair_freq = node.data.local_pairwise_freq  
        items = list(node.data._local_items)       
        idx = {g:i for i,g in enumerate(items)}

        fi = np.array([float(self.noisy_hist[g]) for g in items], dtype=np.float64)
        fi = np.maximum(fi, eps)

        w = np.zeros((K, K), dtype=np.float64)
        for a_i in range(K):
            a_g = items[a_i]
            for b_i in range(a_i + 1, K):
                b_g = items[b_i]
                fij = float(pair_freq.get((a_g, b_g), pair_freq.get((b_g, a_g), eps)))
                fij = max(fij, eps)
                wij = abs(math.log(fij / max(fi[a_i] * fi[b_i], eps)))
                w[a_i, b_i] = wij
                w[b_i, a_i] = wij

        s = np.sum(w, axis=1)          
        start = int(np.argmax(s))      
        chain = [start]
        in_chain = np.zeros(K, dtype=bool)
        in_chain[start] = True
        tail = start

        for _ in range(K - 1):
            cand = np.where(~in_chain)[0]
            if cand.size == 0:
                break
            scores = w[tail, cand]
            j = int(cand[np.argmax(scores)])
            if scores.max() <= 0:
                j = int(cand[np.argmax(s[cand])])
            chain.append(j)
            in_chain[j] = True
            tail = j

        ordered_items = [items[i] for i in chain]  

        hasher = node.data.this_node_hasher
        local_ids = [node.data._g2l[g] for g in ordered_items]
        single_queries = [{lid} for lid in local_ids]
        sig_single = compute_query_sigs_process(hasher=hasher, queries=single_queries)

        sig_single = [np.asarray(sig, dtype=np.int64) for sig in sig_single]
        L = int(sig_single[0].shape[0])

        INF = np.iinfo(np.int64).max
        base_empty = np.full(L, INF, dtype=np.int64)

        prefix_sigs = [None] * (K + 1)
        prefix_sigs[0] = base_empty.copy()
        for b in range(1, K + 1):
            prefix_sigs[b] = np.minimum(prefix_sigs[b - 1], sig_single[b - 1])

        suffix_sigs = [None] * (K + 1)
        suffix_sigs[K] = base_empty.copy()
        for b in range(K - 1, -1, -1):
            suffix_sigs[b] = np.minimum(suffix_sigs[b + 1], sig_single[b])

        e = math.exp(self.per_hash_func_epsilon)
        p = 0.5
        q = 1.0 / (e + 1.0)

        def _p0_from_sig(sig_ids: np.ndarray, subset_size: int) -> float:
            if subset_size <= 0:
                return 1.0
            num_users = len(node.data.perturbed_reports)
            if num_users <= 0:
                return 0.0

            cnt0 = 0
            for this_usr_perturbed_reports in node.data.perturbed_reports:
                matches = 0
                trials = 0
                for (s_idx, perturbed_rep) in this_usr_perturbed_reports:
                    trials += 1
                    y_ell = int(sig_ids[s_idx])
                    if int(perturbed_rep[y_ell]) == 1:
                        matches += 1

                Zbar = matches / trials
                J_hat = (Zbar - q) / (p - q)
                J_hat = min(1.0, max(0.0, J_hat))

                k_idx = int(round(J_hat * subset_size))
                if k_idx <= 0:
                    cnt0 += 1

            return float(cnt0) / float(num_users)

        if hasattr(node.data, "estimated_minhash_hist") and node.data.estimated_minhash_hist is not None:
            p00 = float(np.clip(np.asarray(node.data.estimated_minhash_hist, dtype=np.float64)[0], 0.0, 1.0))
        else:
            p00 = _p0_from_sig(prefix_sigs[K], subset_size=K)

        def _mutual_info_from_p0(p0A: float, p0B: float, p00_: float) -> float:
            p0A = float(np.clip(p0A, 0.0, 1.0))
            p0B = float(np.clip(p0B, 0.0, 1.0))
            p00_ = float(np.clip(p00_, 0.0, 1.0))

            p00v = p00_
            p01v = max(0.0, p0A - p00v)          
            p10v = max(0.0, p0B - p00v)         
            p11v = max(0.0, 1.0 - p00v - p01v - p10v)

            Z = p00v + p01v + p10v + p11v
            if Z <= 0:
                return 0.0
            p00v, p01v, p10v, p11v = [x / Z for x in (p00v, p01v, p10v, p11v)]

            p0a, p1a = p0A, 1.0 - p0A
            p0b, p1b = p0B, 1.0 - p0B

            mi = 0.0
            for pab, pa, pb in [
                (p00v, p0a, p0b),
                (p01v, p0a, p1b),
                (p10v, p1a, p0b),
                (p11v, p1a, p1b),
            ]:
                if pab > eps and pa > eps and pb > eps:
                    mi += pab * math.log(pab / (pa * pb))
            return float(mi)

        best_b = None
        best_mi = None
        best_p0A = None
        best_p0B = None

        for b in range(min_child_size, K - min_child_size + 1):
            p0A = _p0_from_sig(prefix_sigs[b], subset_size=b)
            p0B = _p0_from_sig(suffix_sigs[b], subset_size=K - b)
            mi = _mutual_info_from_p0(p0A, p0B, p00)

            if (best_mi is None) or (mi < best_mi):
                best_mi = mi
                best_b = b
                best_p0A = p0A
                best_p0B = p0B

        if best_b is None:
            return None

        S1 = set(ordered_items[:best_b])
        S2 = set(ordered_items[best_b:])

        meta = {
            "ordered_items": ordered_items,
            "best_b": int(best_b),
            "MI": float(best_mi),
            "p00_full": float(p00),
            "p0A": float(best_p0A),
            "p0B": float(best_p0B),
            "objective": "min I(A;B) using P(X=0) from MinHash+OUE debias",
            "graph_weight": "w_ij = |log(f_ij/(f_i f_j))|",
            "serialization": "greedy chain from max s(i)=sum_j w_ij, then max w_tail,j",
        }
        return S1, S2, meta

    import random

    def _propose_binary_split_random(self, node, min_child_size: int = 20, rng=None):
        items = list(node.data.this_node_item_set)
        n = len(items)

        if n < 2 * min_child_size:
            return None

        if rng is None:
            rng = random

        rng.shuffle(items)

        valid_beg = min_child_size
        valid_end = n - min_child_size    
        if valid_beg >= valid_end:
            return None

        b = rng.randint(valid_beg, valid_end)

        S1 = set(items[:b])
        S2 = set(items[b:])

        meta = {
            "type": "random",
            "boundary": b,
            "num_items": n,
            "size_S1": len(S1),
            "size_S2": len(S2),
        }

        return S1, S2, meta

    
    def _underlying_user_allocation(self):
        num = int(self.user_num * self.alpha) - 1
        self.underlying_users = [0,num]
    
    def _estimate_underly_hist(self, users):
        data = self.data[users[0]:users[1]+1]
        sampled_data = []
        for data_i in data:
            sampled_data.append(random.choice(list(data_i)))
        fo = Frequency_oracle(sampled_data, frequency_oracle_name = 'OUE', epsilon=self.epsilon, domain_size=self.domain_size)
        hist_noisy = fo.get_aggregated_count(smooth=True)

        return hist_noisy
    
    def _estimate_pairwise_cross_node_relations(self, users, users_range, epsilon, only_leaves: bool = False):
        from collections import defaultdict

        l, r = users_range[0], users_range[1]
        if r < l:
            self.cross_node_cofreq = {}
            return self.cross_node_cofreq

        if only_leaves:
            nodes = [n for n in self.tree.all_nodes() if n.is_leaf()]
        else:
            nodes = [n for n in self.tree.all_nodes() if not n.is_root()]
        if len(nodes) < 2:
            self.cross_node_cofreq = {}
            return self.cross_node_cofreq

        leaf_ids = [n.identifier for n in nodes]
        leaf_set_list = [n.data.this_node_item_set for n in nodes]

        item2leaf = {}
        for nid, itemset in zip(leaf_ids, leaf_set_list):
            for it in itemset:
                item2leaf[it] = nid

        leaf_id_to_local = {nid: i for i, nid in enumerate(leaf_ids)}  
        L = len(leaf_ids)
        SPECIAL_NONE_IDX = L  
        data_slice = users[l:r+1]
        sampled_pairs_local = []  
        for S in data_slice:
            touched = set()
            for it in S:
                nid = item2leaf.get(it)
                if nid is not None:
                    touched.add(leaf_id_to_local[nid])

            if len(touched) >= 2:
                a, b = random.sample(list(touched), 2)
                if a > b:
                    a, b = b, a
                sampled_pairs_local.append((a, b))
            else:
                sampled_pairs_local.append((SPECIAL_NONE_IDX, SPECIAL_NONE_IDX))

        fo = Frequency_oracle(
            data=sampled_pairs_local,
            frequency_oracle_name='OUE',
            epsilon=epsilon,             
            domain_size=(L + 1, L + 1),
            merged_domain=None
        )
        
        freq_mat = fo.get_aggregated_frequency(smooth=True)  

        cross_node_cofreq = {}
        for ai in range(L):
            for aj in range(ai + 1, L):
                p = float(freq_mat[ai, aj])
                if p < 0:
                    p = 0.0
                if p == 0.0:
                    continue
                node_i = leaf_ids[ai]
                node_j = leaf_ids[aj]
                cross_node_cofreq[(node_i, node_j)] = p

        self.cross_node_cofreq = cross_node_cofreq

    
    def _determine_split_node(self, node_to_split):
        if len(node_to_split.data.this_node_item_set) >= 50:
            return True
        
        return False
        
    def _partition_items(self, items, fanout):
        lst = list(items)
        n = len(lst)
        size = math.ceil(n / fanout)
        subsets = []
        for i in range(fanout):
            start = i * size
            if start >= n:
                break
            end = min(start + size, n)
            subsets.append(set(lst[start:end]))
        return subsets

    def _set_partitioning_tree_construction(self):
        tree = Tree()
        node_queue = queue.Queue()
        node_id = 0

        tree.create_node(
            tag='root',
            identifier=node_id,
            data=Node_partitioning(this_node_item_set=self.item_set, fanout=self.fanout)
        )
        root = tree.get_node(node_id)
        node_queue.put(root)

        while not node_queue.empty():
            parent = node_queue.get()
            current_set = parent.data.this_node_item_set 

            if len(current_set) > 2 and self._determine_split_node(node_to_split=parent):
                for subset in self._partition_items(current_set, self.fanout):
                    node_id += 1
                    tree.create_node(
                        tag=f'n_{node_id}',
                        identifier=node_id,
                        parent=parent.identifier,
                        data=Node_partitioning(this_node_item_set=subset, fanout=self.fanout)
                    )
                    child = tree.get_node(node_id)
                    node_queue.put(child)

        return tree
    
    def _prepartition_tree_users(self, max_depth_d: int):
        tree_l = self.underlying_users[1] + 1
        tree_r = self.user_num - 1

        if tree_l > tree_r:
            return []

        tree_slice = self.data[tree_l:tree_r+1]
        random.shuffle(tree_slice)
        self.data[tree_l:tree_r+1] = tree_slice

        total = tree_r - tree_l + 1
        if max_depth_d <= 0:
            return [(tree_l, tree_r)]

        B = 2 ** max_depth_d  
        base = max(1, total // B)

        blocks = []
        start = tree_l
        while start <= tree_r and len(blocks) < B:
            end = min(start + base - 1, tree_r)
            blocks.append((start, end))
            start = end + 1

        if start <= tree_r and len(blocks) > 0:
            last_l, last_r = blocks[-1]
            blocks[-1] = (last_l, tree_r)

        return blocks

    def _uniform_user_allocation(self, this_tree):
        tree_user_num = self.user_num - (self.underlying_users[1] - self.underlying_users[0] + 1)
        root = this_tree.get_node(this_tree.root)
        root_subtree_depth = this_tree.subtree(root.identifier).depth() + 1  

        for node in self._bfs(this_tree, return_root=True):
            if node.is_root():
                prop = 1.0 / root_subtree_depth
                node.data.set_allocated_users(self.user_num, tree_user_num, prop)
            else:
                to_be_allocated = this_tree.parent(node.identifier).data.left_user_num
                subtree_depth   = this_tree.subtree(node.identifier).depth() + 1
                prop = 1.0 / subtree_depth
                node.data.set_allocated_users(self.user_num, to_be_allocated, prop)
    
    def _bfs(self, this_tree, return_root = False):
        travl_queue = queue.Queue()
        root = this_tree.get_node(this_tree.root)
        travl_queue.put(root)
        while not travl_queue.empty():
            node = travl_queue.get()
            if not node.is_leaf():
                for child in this_tree.children(node.identifier):
                    travl_queue.put(child)
            if return_root or not node.is_root():
                yield node
    
    def _build_set_partitioning_tree_dynamic(self, max_depth_d: int, min_child_size: int = 500):
        self._underlying_user_allocation()
        self.noisy_hist = self._estimate_underly_hist(self.underlying_users)

        blocks = self._prepartition_tree_users(max_depth_d)
        self._tree_blocks_debug = blocks  

        tree = Tree()
        node_id = 0
        tree.create_node(
            tag="root",
            identifier=node_id,
            data=Node_partitioning(
                this_node_item_set=self.item_set,
                fanout=self.fanout
            )
        )

        q = queue.Queue()
        q.put(tree.get_node(node_id))

        block_cursor = 0
        leaf_nodes = []   
        while not q.empty():
            node = q.get()
            depth = tree.depth(node.identifier)   

            if block_cursor >= len(blocks):
                leaf_nodes.append(node)
                continue

            users_range = blocks[block_cursor]
            block_cursor += 1

            node.data.allocated_users = list(users_range)

            self._estimate_node_with_users(node, users_range, do_pairwise=True)
            t0=time.perf_counter()
            proposal = self._propose_binary_split_by_pmi_serialization_and_MI_cut(
                node,
                min_child_size=min_child_size, 
                if_random_for_ablation_study=False # ablation study for randomly assign items to children nodes
            )
            t1=time.perf_counter()
            if proposal is None:
                var_do_not_split = 0.0
                var_split = float("inf")
            else:
                g_current = max(1, len(tree.all_nodes()) - 1)

                var_do_not_split, var_split = self._evaluate_split_error_by_sampling(
                    node=node,
                    proposal=proposal,
                    g_current=g_current,
                    num_samples=20,     
                    rng=None            
                )

            print('var_split', var_split, 'var_do_not_split', var_do_not_split)

            split_var_flag = (var_split <= var_do_not_split)

            do_split = (
                proposal is not None
                and split_var_flag
                and depth < max_depth_d
            )

            if not do_split:
                leaf_nodes.append(node)
                continue
            S1, S2, meta = proposal
            node_id += 1
            tree.create_node(
                tag=f"n_{node_id}",
                identifier=node_id,
                parent=node.identifier,
                data=Node_partitioning(
                    this_node_item_set=S1,
                    fanout=self.fanout
                )
            )
            q.put(tree.get_node(node_id))

            node_id += 1
            tree.create_node(
                tag=f"n_{node_id}",
                identifier=node_id,
                parent=node.identifier,
                data=Node_partitioning(
                    this_node_item_set=S2,
                    fanout=self.fanout
                )
            )
            q.put(tree.get_node(node_id))

        start_time = time.time()
        self.tree = tree

        self._estimate_pairwise_cross_node_relations(users=self.data, users_range=self.underlying_users, \
            epsilon=self.epsilon, only_leaves=False)
        
        return tree

    def _redistribute_unused_blocks_to_leaves(self, leaf_nodes: List[Node], extra_blocks: List[Tuple[int,int]]):
        if not extra_blocks or not leaf_nodes:
            return

        for leaf in leaf_nodes:
            leaf.data.extra_user_blocks = []

        num_leaves = len(leaf_nodes)
        for idx, br in enumerate(extra_blocks):
            leaf = leaf_nodes[idx % num_leaves]
            leaf.data.extra_user_blocks.append(br)

        for leaf in leaf_nodes:
            extra_bs = getattr(leaf.data, "extra_user_blocks", [])
            if not extra_bs:
                continue
            self._refine_leaf_hist_with_blocks(leaf, extra_bs)
    
    def _refine_leaf_hist_with_blocks(self, leaf: Node, extra_blocks: List[Tuple[int,int]]):
        extra_blocks = list(extra_blocks)
        if not extra_blocks:
            return

        hist_old = getattr(leaf.data, "estimated_jc_hist", None)
        U_old = len(getattr(leaf.data, "perturbed_reports", []))

        if hist_old is None:
            hist_old = None
            U_old = 0
        else:
            hist_old = np.asarray(hist_old, dtype=np.float64)

        old_pairs = getattr(leaf.data, "local_pairwise_freq", None)
        old_hasher = getattr(leaf.data, "this_node_hasher", None)

        total_U = U_old
        combined_hist = hist_old

        for (l, r) in extra_blocks:
            self._estimate_node_with_users(leaf, (l, r), do_pairwise=False)

            hist_new = np.asarray(leaf.data.estimated_jc_hist, dtype=np.float64)
            U_new = len(getattr(leaf.data, "perturbed_reports", []))

            if U_new <= 0:
                continue

            if combined_hist is None:
                combined_hist = hist_new
                total_U = U_new
            else:
                L = max(len(combined_hist), len(hist_new))
                if len(combined_hist) < L:
                    combined_hist = np.pad(combined_hist, (0, L - len(combined_hist)), constant_values=0.0)
                if len(hist_new) < L:
                    hist_new = np.pad(hist_new, (0, L - len(hist_new)), constant_values=0.0)
                combined_hist = (combined_hist * total_U + hist_new * U_new) / (total_U + U_new)
                total_U += U_new

        if combined_hist is not None:
            leaf.data.estimated_jc_hist = combined_hist
            leaf.data.perturbed_reports = [None] * int(total_U)

        if old_pairs is not None:
            leaf.data.local_pairwise_freq = old_pairs
        if old_hasher is not None:
            leaf.data.this_node_hasher = old_hasher

    def _estimate_node_with_users(self, node, users_range, do_pairwise: bool = True):

        node.data._build_node_local_domain()

        if do_pairwise:
            node.data.estimate_local_pairwise_freq(
                user_sets=self.data,
                users_range=users_range,
                epsilon=self.epsilon
            )

        l, r = users_range
        if r < l:
            return

        if self.use_dual_full:
            n = (r - l + 1)
            n_oue = int(round(n * self.dual_split_ratio))
            n_oue = max(0, min(n, n_oue))

            if n_oue > 0:
                range_oue = (l, l + n_oue - 1)
                self._estimate_node_intersection_hist_oue(node, range_oue, eps_oue=self.eps_oue_intersection)
            else:
                node.data.estimated_oue_hist = None
                node.data.num_users_oue = 0
                node.data.eps_oue_intersection = self.eps_oue_intersection

            if n_oue < n:
                range_mh = (l + n_oue, r)
                self._estimate_minhash_results(node=node, users_range=range_mh)
                self._compute_this_node_jc_similarity(node=node)   
            else:
                K = len(node.data.this_node_item_set)
                node.data.estimated_jc_hist = np.zeros(K + 1, dtype=float)
                node.data.estimated_minhash_hist = node.data.estimated_jc_hist
                node.data.num_users_minhash = 0

            return

        self._estimate_minhash_results(node=node, users_range=users_range)
        self._compute_this_node_jc_similarity(node=node)

    def _estimate_minhash_results(self, node, users_range=None):
        if users_range is None:
            l, r = node.data.allocated_users
        else:
            l, r = users_range

        data = self.data[l:r+1]
        L = self.hash_func_maximum_num
        s = self.hash_func_sample_number
        this_node_item_set = node.data.this_node_item_set

        data_local = [node.data._to_local_set(S, node.data._g2l, node.data._other_id) for S in data]

        node.data.this_node_hasher = _MinHasher(
            L=L,
            domain_size=node.data._local_D,
            seed=2025,
            output_domain_size=None
        )

        this_node_user_sig_ids: List[List[int]] = compute_user_sigs_process(
            node.data.this_node_hasher,
            data_local
        )

        node.data.perturbed_reports = []
        for usr_i, data_i in enumerate(data_local):
            rng_u = random.Random((usr_i + 1337) ^ 0x5bf03635)
            sampled_hash_func_idxs = rng_u.sample(range(L), s) if s < L else list(range(L))
            coords_rep = []

            for s_idx in sampled_hash_func_idxs:
                true_cat = this_node_user_sig_ids[usr_i][s_idx]
                if true_cat < 0:
                    continue

                d_out = node.data.this_node_hasher.outD if getattr(
                    node.data.this_node_hasher, "outD", None
                ) is not None else node.data._local_D

                perturbed_rep = _report_oue_value(
                    eps=self.per_hash_func_epsilon,
                    d=d_out,
                    value=true_cat
                )
                coords_rep.append((s_idx, perturbed_rep))

            node.data.perturbed_reports.append(coords_rep)

    def _compute_this_node_jc_similarity(self, node):
        if not hasattr(node.data, '_local_D'):
            node.data._build_node_local_domain()

        node.data._set_node_queries(overall_item_frequency_list=self.noisy_hist)

        queries_local = [node.data._to_local_set(set(q), node.data._g2l, node.data._other_id) for q in node.data.this_node_queries]

        query_sig_ids: List[List[int]] = compute_query_sigs_process(
            hasher=node.data.this_node_hasher,
            queries=queries_local
        )

        e = math.exp(self.per_hash_func_epsilon)
        p = 0.5
        q = 1.0 / (e + 1.0)  

        K = len(node.data.this_node_item_set)
        if K <= 0 or len(query_sig_ids) == 0:
            node.data.estimated_jc_hist = np.zeros(1, dtype=float)
            return

        sig_q_ids_full = query_sig_ids[-1]  

        jc_hist = np.zeros(K + 1, dtype=float)
        node.data.user_x_bins = []   

        num_users = len(node.data.perturbed_reports)
        for this_usr_perturbed_reports in node.data.perturbed_reports:
            matches = 0
            trials  = 0
            for (s_idx, perturbed_rep) in this_usr_perturbed_reports:
                y_ell = sig_q_ids_full[s_idx]
                trials += 1
                if int(perturbed_rep[y_ell]) == 1:
                    matches += 1

            Zbar = matches / trials
            J_hat = (Zbar - q) / (p - q)
            J_hat = min(1.0, max(0.0, J_hat))  

            k_idx = int(round(J_hat * K))
            k_idx = max(0, min(K, k_idx))
            jc_hist[k_idx] += 1.0
            node.data.user_x_bins.append(k_idx)   

        if num_users > 0:
            jc_hist /= float(num_users)

        node.data.estimated_jc_hist = jc_hist
        
        node.data.estimated_minhash_hist = jc_hist
        node.data.num_users_minhash = num_users

    def _estimate_node_intersection_hist_oue(self, node, users_range, eps_oue=None):
        if eps_oue is None:
            eps_oue = self.eps_oue_intersection

        l, r = users_range
        if r < l:
            node.data.estimated_oue_hist = None
            node.data.num_users_oue = 0
            node.data.eps_oue_intersection = eps_oue
            return None

        S_node = node.data.this_node_item_set
        K = len(S_node)
        if K <= 0:
            node.data.estimated_oue_hist = np.zeros(1, dtype=float)
            node.data.num_users_oue = (r - l + 1)
            node.data.eps_oue_intersection = eps_oue
            return node.data.estimated_oue_hist

        xs = []
        for S_u in self.data[l:r+1]:
            x = 0
            x = len(S_u.intersection(S_node))
            if x < 0: x = 0
            if x > K: x = K
            xs.append(int(x))

        fo = Frequency_oracle(
            data=xs,
            frequency_oracle_name='OUE',
            epsilon=eps_oue,
            domain_size=K + 1,
            merged_domain=None
        )
        hist = fo.get_aggregated_frequency(smooth=True)  

        hist = np.asarray(hist, dtype=np.float64)
        if len(hist) < K + 1:
            hist = np.pad(hist, (0, K + 1 - len(hist)), constant_values=0.0)
        elif len(hist) > K + 1:
            hist = hist[:K + 1]

        hist = np.maximum(hist, 0.0)
        s = float(hist.sum())
        if s > 0:
            hist /= s

        node.data.estimated_oue_hist = hist
        node.data.num_users_oue = len(xs)
        node.data.eps_oue_intersection = eps_oue
        return hist

    def _mix_from_hist_any(self, node, Q_items: Set[int], t: int, hist: np.ndarray,
                           extra_cross_pairs=None, use_beta_binom=True) -> float:
        if hist is None:
            return 0.0
        K = len(node.data.this_node_item_set)
        hist = np.asarray(hist, dtype=np.float64)
        if len(hist) < K + 1:
            hist = np.pad(hist, (0, K + 1 - len(hist)), constant_values=0.0)
        elif len(hist) > K + 1:
            hist = hist[:K + 1]

        alpha_vec = self._alpha_tail_vector(
            node, Q_items, int(t),
            extra_cross_pairs=extra_cross_pairs,
            use_beta_binom=use_beta_binom
        )
        f_hat = float(np.dot(hist, alpha_vec))
        return max(0.0, min(1.0, f_hat))

    def _bb_var_from_hist_any(self, node, Q_items: Set[int], t: int, hist: np.ndarray, Un: int,
                             extra_cross_pairs=None, use_beta_binom=True) -> float:
        if hist is None or Un <= 0:
            return float("inf")
        K = len(node.data.this_node_item_set)
        hist = np.asarray(hist, dtype=np.float64)
        if len(hist) < K + 1:
            hist = np.pad(hist, (0, K + 1 - len(hist)), constant_values=0.0)
        elif len(hist) > K + 1:
            hist = hist[:K + 1]

        alpha_vec = self._alpha_tail_vector(
            node, Q_items, int(t),
            extra_cross_pairs=extra_cross_pairs,
            use_beta_binom=use_beta_binom
        )
        f_hat = float(np.dot(hist, alpha_vec))
        sum_w2H = float(np.dot(hist, alpha_vec ** 2))
        var = (sum_w2H - f_hat ** 2) / float(max(1, Un))
        return max(0.0, float(var))

    def _oue_var_from_hist(self, node, Q_items: Set[int], t: int, hist_oue: np.ndarray, Un: int, eps_oue: float) -> float:
        if hist_oue is None or Un <= 0:
            return float("inf")

        alpha_vec = self._alpha_tail_vector(
            node, Q_items, int(t),
            extra_cross_pairs=None,
            use_beta_binom=True
        )
        alpha_vec = np.asarray(alpha_vec, dtype=np.float64)

        e = math.exp(eps_oue)
        denom = (e - 1.0) ** 2
        if denom <= 0:
            return float("inf")

        var_bin = (4.0 * e) / (denom * float(Un))   
        
        return max(0.0, var_bin)

    def _evaluate_split_error_by_sampling(
        self,
        node,
        proposal,
        g_current: int,
        num_samples: int = 20,
        rng: Optional[random.Random] = None,
    ) -> Tuple[float, float]:
        if rng is None:
            rng = random.Random(2025 + node.identifier)

        S1, S2, meta = proposal
        S_union = set(S1).union(S2)
        items_union = list(S_union)
        m = len(items_union)
        if m < 3:
            return 0.0, float("inf")

        g_current = max(1, int(g_current))

        sum_no = 0.0
        sum_split = 0.0
        cnt = 0

        for _ in range(num_samples):
            s_upper = min(m, 500)
            s = rng.randint(3, s_upper)
            Q_items = set(rng.sample(items_union, s))
            if s <= 2:
                continue
            if s == 3:
                t = 2
            else:
                t = rng.randint(2, s - 1)

            Pt = self._approx_tail_with_pairs(
                items_set=Q_items,
                t=t,
                node_for_pairs=node,    
                extra_cross_pairs=None
            )
            Pt = max(0.0, min(1.0, float(Pt)))
            var1_no = self._global_tail_noise_sampling_mse(
                Pt=Pt,
                t=t,
                g=g_current
            )
            var2_no = self._node_reconstruction_variance(
                node=node,
                query_subset=Q_items,
                t=t,
                use_beta_binom=False
            )
            err_no = var1_no + var2_no

            var1_split = self._global_tail_noise_sampling_mse(
                Pt=Pt,
                t=t,
                g=g_current + 1    
            )

            Q1 = Q_items.intersection(S1)
            Q2 = Q_items.intersection(S2)

            var2_child1 = 0.0
            if len(Q1) > 0:
                var2_child1 = self._node_reconstruction_variance(
                    node=node,         
                    query_subset=Q1,
                    t=min(t, len(Q1)),   
                    use_beta_binom=False
                )

            var2_child2 = 0.0
            if len(Q2) > 0:
                var2_child2 = self._node_reconstruction_variance(
                    node=node,
                    query_subset=Q2,
                    t=min(t, len(Q2)),
                    use_beta_binom=False
                )

            if len(Q1) > 0 and len(Q2) > 0:
                var2_split = 0.5 * (var2_child1 + var2_child2)
            elif len(Q1) > 0:
                var2_split = var2_child1
            elif len(Q2) > 0:
                var2_split = var2_child2
            else:
                var2_split = 0.0

            err_split = var1_split + var2_split

            sum_no += err_no
            sum_split += err_split
            cnt += 1

        if cnt == 0:
            return 0.0, float("inf")

        return sum_no / cnt, sum_split / cnt

    def _global_tail_noise_sampling_mse(self, Pt: float, t: int, g: int,
                                        eps_ldp: Optional[float] = None) -> float:
        Pt = float(max(0.0, min(1.0, Pt)))
        if g <= 0 or self.user_num <= 0:
            return 0.0

        U = float(self.user_num)
        g = float(g)

        if eps_ldp is None:
            eps_ldp = self.per_hash_func_epsilon

        e = math.exp(eps_ldp)
        p = 0.5
        q = 1.0 / (e + 1.0)
        q1, q0 = p, q

        denom = (q1 - q0) ** 2
        if denom <= 0:
            return 0.0

        sampling_term = (1.0 - 1.0 / g) * Pt * (1.0 - Pt) * g / U
        noise_term = (g / U) * (
            Pt * q1 * (1.0 - q1) / denom
            + (1.0 - Pt) * q0 * (1.0 - q0) / denom
        )

        mse = sampling_term + noise_term
        return max(0.0, float(mse))

    def _node_bb_variance_from_hist(self, node, Q_items: Set[int], t: int,
                                extra_cross_pairs: Optional[Dict[Tuple[int,int], float]] = None,
                                use_beta_binom: bool = True) -> float:
        hist = getattr(node.data, "estimated_jc_hist", None)
        if hist is None:
            return 0.0

        K = len(node.data.this_node_item_set)
        if K <= 0:
            return 0.0

        hist = np.asarray(hist, dtype=np.float64)
        if len(hist) < K + 1:
            hist = np.pad(hist, (0, K + 1 - len(hist)), constant_values=0.0)
        elif len(hist) > K + 1:
            hist = hist[:K + 1]

        Un = len(getattr(node.data, "perturbed_reports", []))
        if Un <= 0:
            return 0.0

        alpha_vec = self._alpha_tail_vector(node, Q_items, int(t),
                                            extra_cross_pairs=extra_cross_pairs,
                                            use_beta_binom=use_beta_binom)

        f_hat_full = float(np.dot(hist, alpha_vec))
        sum_w2H    = float(np.dot(hist, alpha_vec ** 2))

        var = (sum_w2H - f_hat_full ** 2) / float(Un)
        return max(0.0, float(var))

    def _node_reconstruction_variance_old(self, node, query_subset: Set[int], t: int, use_beta_binom: bool = True) -> float:
        S_n = node.data.this_node_item_set
        R = S_n.intersection(query_subset)

        if len(R) == 0:
            return 0.0

        t_int = int(t)

        if len(R) < t_int:
            Pt_local = self._approx_tail_with_pairs(R, t_int, node_for_pairs=node, extra_cross_pairs=None)
            return float(Pt_local ** 2)

        return self._node_bb_variance_from_hist(node, R, t_int, extra_cross_pairs=None, use_beta_binom=use_beta_binom)

    def _node_reconstruction_variance(self, node, query_subset: Set[int], t: int, use_beta_binom: bool = True) -> float:
        S_n = node.data.this_node_item_set
        R = S_n.intersection(query_subset)

        if len(R) == 0:
            return 0.0

        t_int = int(t)

        if not use_beta_binom:
            U = self.underlying_users[1] - self.underlying_users[0] + 1
            U = max(1, U)

            ps = []
            for i in R:
                pi = float(self.noisy_hist[i]) / U
                if pi < 0.0:
                    pi = 0.0
                if pi > 1.0:
                    pi = 1.0
                ps.append(pi)

            if not ps:
                return 0.0

            mu = sum(ps)
            var = sum(p * (1.0 - p) for p in ps)

            if var <= 1e-9:
                Pt_local = 1.0 if mu >= t_int else 0.0
            else:
                z = (t_int - 0.5 - mu) / math.sqrt(var)
                Pt_local = _normal_ccdf(z)

            Pt_local = max(0.0, min(1.0, float(Pt_local)))
            return float(Pt_local ** 2)

        if len(R) < t_int:
            Pt_local = self._approx_tail_with_pairs(
                R,
                t_int,
                node_for_pairs=node,
                extra_cross_pairs=None
            )
            Pt_local = max(0.0, min(1.0, float(Pt_local)))
            return float(Pt_local ** 2)

        return self._node_bb_variance_from_hist(
            node,
            R,
            t_int,
            extra_cross_pairs=None,
            use_beta_binom=True  
        )

    def compute_partial_threshold_freq(self, node, intersect_set, query_threshold):
        sub_ans = 0.0
        for k in range(0, len(node.data.this_node_item_set) + 1):
            pk = float(node.data.estimated_jc_hist[k])
            if pk == 0.0:
                continue

            max_x = min(k, len(intersect_set))
            if int(query_threshold) > max_x:
                continue

            tail = 0.0
            for thre_val in range(int(query_threshold), max_x + 1):
                tail += math.comb(len(intersect_set), thre_val) * math.comb(len(node.data.this_node_item_set) - len(intersect_set), k - thre_val) / math.comb(len(node.data.this_node_item_set), k)

            sub_ans += pk * tail

        return sub_ans

    
    def compute_threshold_freq(self, node, query_threshold):
        sub_ans = sum(node.data.estimated_jc_hist[max(0, int(query_threshold)):]) 
        
        return sub_ans
    
    def _approx_tail_with_pairs(self, items_set, t, node_for_pairs=None, extra_cross_pairs=None):
        items = sorted(list(items_set))
        if len(items) == 0:
            return 0.0
        t = int(t)
        if t <= 0:
            return 1.0

        U = self.underlying_users[1] - self.underlying_users[0] + 1
        p = {i: max(0.0, min(1.0, float(self.noisy_hist[i]) / max(1, U))) for i in items}

        local_pairs = {}
        if node_for_pairs is not None and hasattr(node_for_pairs.data, "local_pairwise_freq"):
            for (a, b), v in node_for_pairs.data.local_pairwise_freq.items():
                if a == -1 and b == -1:
                    continue
                if a in p and b in p:
                    local_pairs[(a, b)] = float(v)

        if extra_cross_pairs:
            for (a, b), v in extra_cross_pairs.items():
                if a in p and b in p:
                    local_pairs.setdefault((a, b), float(v))

        mu = sum(p.values())
        var = 0.0
        for i in items:
            var += p[i] * (1.0 - p[i])
        for idx_i in range(len(items)):
            i = items[idx_i]
            for idx_j in range(idx_i + 1, len(items)):
                j = items[idx_j]
                fij = local_pairs.get((i, j), None)
                if fij is None:
                    fij = local_pairs.get((j, i), None)
                if fij is None:
                    fij = p[i] * p[j]
                var += 2.0 * (fij - p[i] * p[j])


        var = max(var, 1e-9)
        z = (t - 0.5 - mu) / math.sqrt(var)
        return _normal_ccdf(z)

    def _node_partial_tail(self, node, intersect_set, t):
        if len(intersect_set) == 0:
            return 0.0
        return self._approx_tail_with_pairs(intersect_set, t, node_for_pairs=node, extra_cross_pairs=None)

    def _synthesize_cross_pairs(self, nodeA, nodeB, subsetA, subsetB):
        if not hasattr(self, "cross_node_cofreq"):
            return {}

        idA, idB = nodeA.identifier, nodeB.identifier
        if idA > idB:
            idA, idB = idB, idA
            subsetA, subsetB = subsetB, subsetA
            nodeA, nodeB = nodeB, nodeA

        P_AB = float(self.cross_node_cofreq.get((idA, idB), 0.0))
        if P_AB <= 0.0:
            return {}

        U = self.underlying_users[1] - self.underlying_users[0] + 1
        wA = {i: max(0.0, min(1.0, float(self.noisy_hist[i]) / max(1, U))) for i in subsetA}
        wB = {j: max(0.0, min(1.0, float(self.noisy_hist[j]) / max(1, U))) for j in subsetB}
        sA = sum(wA.values()); sB = sum(wB.values())
        if sA <= 0.0 or sB <= 0.0:
            return {}

        cross_pairs = {}
        for i in subsetA:
            for j in subsetB:
                cross_pairs[(i, j)] = P_AB * (wA[i] / sA) * (wB[j] / sB)
        return cross_pairs


    def query(self, query_subset, query_threshold, set_partitioning_tree=None):
        if set_partitioning_tree is None:
            set_partitioning_tree = self.tree

        t = int(query_threshold)
        Q = set(query_subset)

        full_node, partial_nodes = self._select_nodes_for_query(Q, t, set_partitioning_tree)

        f_full, var_full = self._estimate_full_component(full_node, Q, t)

        f_partial, var_partial = self._estimate_partial_component(partial_nodes, Q, t)

        if full_node is None or not np.isfinite(var_full) or var_full <= 0.0:
            return max(0.0, min(1.0, f_partial))

        if not partial_nodes or not np.isfinite(var_partial) or var_partial <= 0.0:
            return max(0.0, min(1.0, f_full))

        Var1 = float(var_full)
        Var2 = float(var_partial)

        denom = Var1 + Var2
        if denom <= 0.0:
            f_hat = 0.5 * (f_full + f_partial)
        else:
            w_full    = Var2 / denom    
            w_partial = Var1 / denom    
            f_hat = w_full * f_full + w_partial * f_partial

        f_hat = max(0.0, min(1.0, f_hat))
        return f_hat

    def __intersection(self, this_node_item_set, query_subset, query_threshold):
        intersect_set = this_node_item_set.intersection(query_subset)
        if len(intersect_set) < query_threshold:
            return -1, intersect_set
        elif len(intersect_set)==len(query_subset):
            if len(this_node_item_set)==len(query_subset):
                return 0, intersect_set
            else:
                return 2, intersect_set
        else: 
            return 1, intersect_set
    
    def _select_nodes_for_query(self, Q: Set[int], t: int, tree=None):
        if tree is None:
            tree = self.tree

        full_cover_node = None
        max_depth = -1
        for node in tree.all_nodes():
            S = node.data.this_node_item_set
            if Q.issubset(S):
                d = tree.depth(node.identifier)
                if d > max_depth:
                    max_depth = d
                    full_cover_node = node

        partial_nodes: List[Tuple[Node, Set[int]]] = []
        covered_items: Set[int] = set()

        nodes_sorted = sorted(
            tree.all_nodes(),
            key=lambda n: tree.depth(n.identifier),
            reverse=True
        )

        for node in nodes_sorted:
            if node.is_root():
                continue
            if full_cover_node is not None and node.identifier == full_cover_node.identifier:
                continue

            S = node.data.this_node_item_set
            R = S.intersection(Q)

            if len(R) < t:
                continue
            if Q.issubset(S):
                continue

            if R & covered_items:
                continue

            partial_nodes.append((node, R))
            covered_items |= R

            if covered_items.issuperset(Q):
                break

        return full_cover_node, partial_nodes

    def _alpha_tail_vector_gauss(self, node, Q_items: Set[int], t: int, \
        extra_cross_pairs: Optional[Dict[Tuple[int,int], float]] = None, use_beta_binom: bool = True):
        K = len(node.data.this_node_item_set)
        t = int(t)

        if K < 0:
            return np.zeros(0, dtype=np.float64)

        if t <= 0:
            return np.ones(K + 1, dtype=np.float64)

        W_Q, rho = self._estimate_W_and_rho(node, Q_items, extra_cross_pairs)

        W_Q = float(min(1.0, max(0.0, W_Q)))
        rho = float(max(0.0, rho))

        if W_Q <= 0.0:
            return np.zeros(K + 1, dtype=np.float64)

        if W_Q >= 1.0:
            v = np.zeros(K + 1, dtype=np.float64)
            v[t:] = 1.0 if t <= K else 0.0
            return v

        if not use_beta_binom:
            rho = 0.0

        alpha_vec = np.zeros(K + 1, dtype=np.float64)

        for k in range(0, K + 1):
            if k < t:
                alpha_vec[k] = 0.0
                continue
            mu = k * W_Q
            factor = 1.0 + (k - 1) * rho
            if factor < 1e-9:
                factor = 1e-9
            var = k * W_Q * (1.0 - W_Q) * factor
            a = _gaussian_tail(mu, var, t)

            if a < 0.0: a = 0.0
            if a > 1.0: a = 1.0
            alpha_vec[k] = a

        return alpha_vec
    
    def _alpha_tail_vector_mh(self, node, Q_items: Set[int], t: int):
        cache = getattr(node.data, "_alpha_mh_cache", None)
        if cache is None:
            node.data._alpha_mh_cache = {}
            cache = node.data._alpha_mh_cache
        key = (hash(tuple(sorted(Q_items))), len(Q_items), int(t))
        if key in cache:
            return cache[key]
        
        K = len(node.data.this_node_item_set)
        t = int(t)
        if K <= 0:
            return np.zeros(1, dtype=np.float64)

        reps = getattr(node.data, "perturbed_reports", None) or []
        xbins = getattr(node.data, "user_x_bins", None)

        if (not reps) or (xbins is None) or (len(xbins) != len(reps)):
            print("mh learn fail:",
      "reps", len(reps),
      "xbins_is_none", (xbins is None),
      "xbins_len", (None if xbins is None else len(xbins)))

            return None

        s = len(Q_items)
        if s <= 0:
            return np.zeros(K + 1, dtype=np.float64)

        if not hasattr(node.data, "_local_D"):
            node.data._build_node_local_domain()
        q_local = node.data._to_local_set(set(Q_items), node.data._g2l, node.data._other_id)
        sig_q = compute_query_sigs_process(hasher=node.data.this_node_hasher, queries=[q_local])[0]
        sig_q = np.asarray(sig_q, dtype=np.int64)

        e = math.exp(self.per_hash_func_epsilon)
        p = 0.5
        q = 1.0 / (e + 1.0)

        hit_cnt = np.zeros(K + 1, dtype=np.float64)
        tot_cnt = np.zeros(K + 1, dtype=np.float64)

        for u, usr_rep in enumerate(reps):
            k = int(xbins[u])
            if k < 0: 
                continue
            if k > K:
                k = K

            tot_cnt[k] += 1.0

            matches = 0
            trials = 0
            for (s_idx, perturbed_rep) in usr_rep:
                y = sig_q[s_idx]
                trials += 1
                if int(perturbed_rep[y]) == 1:
                    matches += 1

            if trials <= 0:
                continue

            Zbar = matches / trials
            J_hat = (Zbar - q) / (p - q)
            J_hat = min(1.0, max(0.0, J_hat))

            if J_hat <= 0.0:
                x_hat = 0.0
            else:
                x_hat = (J_hat * (k + s)) / (1.0 + J_hat)

            x_hat = max(0.0, min(float(min(k, s)), float(x_hat)))

            if x_hat >= t:
                hit_cnt[k] += 1.0

        alpha_mh = np.zeros(K + 1, dtype=np.float64)
        mask = tot_cnt > 0
        alpha_mh[mask] = hit_cnt[mask] / tot_cnt[mask]

        alpha_mh[:min(t, K+1)] = 0.0
        alpha_mh = np.clip(alpha_mh, 0.0, 1.0)
        cache[key] = alpha_mh
        return alpha_mh
    
    def _alpha_tail_vector(self, node, Q_items: Set[int], t: int,
                        extra_cross_pairs=None, use_beta_binom: bool = True):
        eta = float(getattr(self, "alpha_eta", 0.5))

        alpha_mh = self._alpha_tail_vector_mh(node, Q_items, int(t))

        if eta == 1:
            assert alpha_mh is not None
            if alpha_mh is not None:
                return alpha_mh
            return self._alpha_tail_vector_gauss(node, Q_items, int(t),
                                                extra_cross_pairs=extra_cross_pairs,
                                                use_beta_binom=use_beta_binom)

        alpha_gauss = self._alpha_tail_vector_gauss(node, Q_items, int(t),
                                                    extra_cross_pairs=extra_cross_pairs,
                                                    use_beta_binom=use_beta_binom)

        if alpha_mh is None:
            return alpha_gauss

        alpha = (1.0 - eta) * np.asarray(alpha_gauss, dtype=np.float64) + eta * np.asarray(alpha_mh, dtype=np.float64)
        alpha = np.clip(alpha, 0.0, 1.0)
        return alpha
    
    def _estimate_full_component(self, node, Q: Set[int], t: int):
        if node is None:
            return 0.0, float("inf")

        def _single_from_current():
            f_full = self._mix_from_node_hist(node, Q, int(t))
            var_full = self._node_bb_variance_from_hist(node, Q, int(t), extra_cross_pairs=None)
            f_full = max(0.0, min(1.0, float(f_full)))
            var_full = max(0.0, float(var_full))
            if not getattr(node.data, "perturbed_reports", None):
                var_full = float("inf")
            return f_full, var_full

        if not self.use_dual_full:
            return _single_from_current()

        hist_mh  = getattr(node.data, "estimated_minhash_hist", None)
        Un_mh    = int(getattr(node.data, "num_users_minhash", 0))
        hist_oue = getattr(node.data, "estimated_oue_hist", None)
        Un_oue   = int(getattr(node.data, "num_users_oue", 0))
        eps_oue  = float(getattr(node.data, "eps_oue_intersection", self.eps_oue_intersection))

        f_mh = self._mix_from_hist_any(node, Q, int(t), hist_mh)
        var_mh = self._bb_var_from_hist_any(node, Q, int(t), hist_mh, Un_mh)

        f_oue = self._mix_from_hist_any(node, Q, int(t), hist_oue)
        var_oue = self._oue_var_from_hist(node, Q, int(t), hist_oue, Un_oue, eps_oue)

        if not np.isfinite(var_mh) or var_mh <= 0.0:
            return max(0.0, min(1.0, float(f_oue))), float(var_oue)
        if not np.isfinite(var_oue) or var_oue <= 0.0:
            return max(0.0, min(1.0, float(f_mh))), float(var_mh)

        denom = var_oue + var_mh
        if denom <= 0.0:
            f_full = 0.5 * (f_mh + f_oue)
            var_full = min(var_mh, var_oue)
        else:
            f_full = (var_oue * f_mh + var_mh * f_oue) / denom
            var_full = (var_oue * var_mh) / denom

        f_full = max(0.0, min(1.0, float(f_full)))
        var_full = max(0.0, float(var_full))
        return f_full, var_full
    
    def _get_node_pxk_hist(self, node, Q_items: Optional[Set[int]] = None, t: Optional[int] = None, \
        extra_cross_pairs=None, use_beta_binom: bool = True,) -> np.ndarray:
        K = len(node.data.this_node_item_set)

        hist_m = getattr(node.data, "estimated_jc_hist", None)
        if hist_m is None:
            return None

        hist_m = np.asarray(hist_m, dtype=np.float64)
        if len(hist_m) < K + 1:
            hist_m = np.pad(hist_m, (0, K + 1 - len(hist_m)), constant_values=0.0)
        elif len(hist_m) > K + 1:
            hist_m = hist_m[:K + 1]

        sm = float(hist_m.sum())
        if sm > 0:
            hist_m = hist_m / sm

        if not getattr(self, "use_hybrid_pxk_in_partial", False):
            return hist_m

        hist_o = getattr(node.data, "estimated_jc_hist_oue", None)
        if hist_o is None:
            hist_o = getattr(node.data, "estimated_oue_hist", None)
        if hist_o is None:
            return hist_m

        hist_o = np.asarray(hist_o, dtype=np.float64)
        if len(hist_o) < K + 1:
            hist_o = np.pad(hist_o, (0, K + 1 - len(hist_o)), constant_values=0.0)
        elif len(hist_o) > K + 1:
            hist_o = hist_o[:K + 1]

        so = float(hist_o.sum())
        if so > 0:
            hist_o = hist_o / so
        else:
            return hist_m

        w_fallback = getattr(self, "hybrid_pxk_weight", 0.5)
        if callable(w_fallback):
            w_fallback = float(w_fallback(node))
        w_fallback = float(max(0.0, min(1.0, w_fallback)))

        t = int(t)

        Un_mh = int(getattr(node.data, "num_users_minhash", 0))
        if Un_mh <= 0:
            Un_mh = len(getattr(node.data, "perturbed_reports", []) or [])

        Un_oue = int(getattr(node.data, "num_users_oue", 0))
        if Un_oue <= 0:
            Un_oue = int(getattr(node.data, "num_users_intersection_oue", 0))

        eps_oue = float(getattr(node.data, "eps_oue_intersection",
                                getattr(self, "eps_oue_intersection", self.epsilon)))

        var_mh = self._bb_var_from_hist_any(
            node=node,
            Q_items=Q_items,
            t=t,
            hist=hist_m,
            Un=Un_mh,
            extra_cross_pairs=extra_cross_pairs,
            use_beta_binom=use_beta_binom
        )
        var_oue = self._oue_var_from_hist(
            node=node,
            Q_items=Q_items,
            t=t,
            hist_oue=hist_o,
            Un=Un_oue,
            eps_oue=eps_oue
        )

        if (not np.isfinite(var_mh)) or var_mh <= 0.0:
            w_mh = 0.0
        elif (not np.isfinite(var_oue)) or var_oue <= 0.0:
            w_mh = 1.0
        else:
            denom = float(var_oue + var_mh)
            if denom <= 0.0:
                w_mh = w_fallback
            else:
                w_mh = float(var_oue / denom)

        w_mh = float(max(0.0, min(1.0, w_mh)))

        hist_mix = w_mh * hist_m + (1.0 - w_mh) * hist_o

        hist_mix = np.maximum(hist_mix, 0.0)
        s = float(hist_mix.sum())
        if s > 0:
            hist_mix = hist_mix / s
        else:
            hist_mix = hist_m

        return hist_mix



    def _estimate_partial_component(self, partial_nodes: List[Tuple[Node, Set[int]]], Q: Set[int], t: int):
        if not partial_nodes:
            return 0.0, float("inf")

        t = int(t)

        f_intra = 0.0
        var_intra = 0.0
        intra_infos = []   

        for node, R in partial_nodes:
            if not R:
                continue

            hist_used = self._get_node_pxk_hist(node, Q_items=R, t=t, extra_cross_pairs=None, use_beta_binom=True)

            fi = self._mix_from_hist_any(node, R, t, hist=hist_used,
                                        extra_cross_pairs=None, use_beta_binom=True)
            fi = max(0.0, min(1.0, float(fi)))

            Un = len(getattr(node.data, "perturbed_reports", []) or [])
            vi = self._bb_var_from_hist_any(node, R, t, hist=hist_used, Un=Un,
                                            extra_cross_pairs=None, use_beta_binom=True)
            vi = max(0.0, float(vi))

            f_intra += fi
            var_intra += vi
            intra_infos.append((node, R, fi, vi))

        if not intra_infos:
            return 0.0, float("inf")
        f_cross = 0.0
        var_cross = 0.0

        U_total = max(1, int(self.user_num))
        cross_node_freq = getattr(self, "cross_node_cofreq", {})

        for i in range(len(intra_infos)):
            node_u, Ru, fi, _ = intra_infos[i]
            for j in range(i + 1, len(intra_infos)):
                node_v, Rv, fj, _ = intra_infos[j]

                if not Ru or not Rv:
                    continue

                S_union = Ru.union(Rv)
                if len(S_union) < t:
                    continue

                id_u = node_u.identifier
                id_v = node_v.identifier
                key_uv = (id_u, id_v) if id_u < id_v else (id_v, id_u)
                p_uv = float(cross_node_freq.get(key_uv, 0.0))
                if p_uv <= 0.0:
                    continue

                cross_pairs = self._synthesize_cross_pairs(node_u, node_v, Ru, Rv)

                pairs_uv_all: Dict[Tuple[int,int], float] = {}

                if hasattr(node_u.data, "local_pairwise_freq"):
                    for (a, b), v in node_u.data.local_pairwise_freq.items():
                        if a == -1 or b == -1:
                            continue
                        if a not in S_union or b not in S_union:
                            continue
                        key = (a, b) if a < b else (b, a)
                        pairs_uv_all[key] = float(v)

                if hasattr(node_v.data, "local_pairwise_freq"):
                    for (a, b), v in node_v.data.local_pairwise_freq.items():
                        if a == -1 or b == -1:
                            continue
                        if a not in S_union or b not in S_union:
                            continue
                        key = (a, b) if a < b else (b, a)
                        pairs_uv_all[key] = float(v)

                for (a, b), v in cross_pairs.items():
                    if a not in S_union or b not in S_union:
                        continue
                    key = (a, b) if a < b else (b, a)
                    pairs_uv_all[key] = pairs_uv_all.get(key, 0.0) + float(v)

                f_uv = self._approx_tail_with_pairs(
                    items_set=S_union,
                    t=t,
                    node_for_pairs=None,          
                    extra_cross_pairs=pairs_uv_all
                )
                f_uv = max(0.0, min(1.0, float(f_uv)))
                f_cross += f_uv

                var_p = p_uv * (1.0 - p_uv) / float(U_total)
                if p_uv > 0.0 and var_p > 0.0:
                    gprime = f_uv / max(p_uv, 1e-9)
                    var_cross += (gprime ** 2) * var_p

        pairs_all = self._build_global_pairs_for_Q(partial_nodes, Q)

        items_all: Set[int] = set()
        for _, R in partial_nodes:
            items_all |= R
        if not items_all:
            items_all = set(Q)

        Pt_best = self._approx_tail_with_pairs(
            items_set=items_all,
            t=t,
            node_for_pairs=None,
            extra_cross_pairs=pairs_all
        )
        Pt_best = max(0.0, min(1.0, float(Pt_best)))

        f_partial_mean = f_intra + f_cross
        f0 = Pt_best - f_partial_mean
        if f0 < 0.0:
            f0 = 0.0

        f_partial = f_partial_mean + f0
        f_partial = max(0.0, min(1.0, f_partial))

        var0 = f0 * (1.0 - f0) / float(U_total)

        var_partial = var_intra + var_cross + var0
        if var_partial <= 0.0:
            var_partial = 1e-12

        return f_partial, var_partial


    def _build_global_pairs_for_Q(self, partial_nodes: List[Tuple[Node, Set[int]]], Q: Set[int]) -> Dict[Tuple[int,int], float]:
        pairs: Dict[Tuple[int,int], float] = {}

        for node, R in partial_nodes:
            if not hasattr(node.data, "local_pairwise_freq"):
                continue
            for (a, b), v in node.data.local_pairwise_freq.items():
                if a == -1 or b == -1:
                    continue
                if a not in Q or b not in Q:
                    continue
                key = (a, b) if a < b else (b, a)
                pairs[key] = float(v)

        if hasattr(self, "cross_node_cofreq"):
            for i in range(len(partial_nodes)):
                node_u, Ru = partial_nodes[i]
                for j in range(i + 1, len(partial_nodes)):
                    node_v, Rv = partial_nodes[j]
                    if not Ru or not Rv:
                        continue
                    cross_pairs_uv = self._synthesize_cross_pairs(node_u, node_v, Ru, Rv)
                    for (a, b), v in cross_pairs_uv.items():
                        if a not in Q or b not in Q:
                            continue
                        key = (a, b) if a < b else (b, a)
                        pairs[key] = pairs.get(key, 0.0) + float(v)

        return pairs

    def _estimate_W_and_rho(self, node, Q_items: Set[int], extra_cross_pairs: Optional[Dict[Tuple[int,int], float]] = None):
        U = self.underlying_users[1] - self.underlying_users[0] + 1
        p_node = {i: max(0.0, min(1.0, float(self.noisy_hist[i]) / max(1, U))) for i in node.data.this_node_item_set}
        sum_p_node = sum(p_node.values()) + 1e-12

        Qn = [i for i in Q_items if i in p_node]
        if not Qn:
            return 0.0, 0.0  

        W_Q = (sum(p_node[i] for i in Qn)) / sum_p_node
        W_Q = min(1.0, max(0.0, W_Q))

        var_Q = 0.0
        for i in Qn:
            var_Q += p_node[i] * (1.0 - p_node[i])

        local_pairs = {}
        if hasattr(node.data, "local_pairwise_freq"):
            for (a, b), v in node.data.local_pairwise_freq.items():
                if a == -1 and b == -1:
                    continue
                local_pairs[(a, b)] = float(v)

        cov_extra = 0.0
        for idx_i in range(len(Qn)):
            i = Qn[idx_i]
            for idx_j in range(idx_i + 1, len(Qn)):
                j = Qn[idx_j]
                fij = None
                if (i, j) in local_pairs:
                    fij = local_pairs[(i, j)]
                elif (j, i) in local_pairs:
                    fij = local_pairs[(j, i)]
                elif extra_cross_pairs is not None:
                    fij = extra_cross_pairs.get((i, j), extra_cross_pairs.get((j, i), None))
                if fij is None:
                    fij = p_node[i] * p_node[j]
                cov_extra += (fij - p_node[i] * p_node[j])

        var_Q += 2.0 * cov_extra
        var_Q = max(var_Q, 1e-9)

        base_var_Q = sum(p_node[i] * (1.0 - p_node[i]) for i in Qn) + 1e-12
        overdisp_unit = max(0.0, (var_Q - base_var_Q) / base_var_Q)

        rho = min(0.2, max(0.0, 0.5 * overdisp_unit))
        return W_Q, rho

    def _alpha_tail_given_k(self, node, Q_items: Set[int], t: int, k: int,
                    extra_cross_pairs: Optional[Dict[Tuple[int,int], float]] = None,
                    use_beta_binom: bool = True) -> float:
        t = int(t)
        k = int(k)
        if t <= 0:
            return 1.0
        if k < t:
            return 0.0
        if k <= 0:
            return 0.0

        W_Q, rho = self._estimate_W_and_rho(node, Q_items, extra_cross_pairs)
        W_Q = float(min(1.0, max(0.0, W_Q)))
        rho = float(max(0.0, rho))
        if W_Q <= 0.0:
            return 0.0
        if W_Q >= 1.0:
            return 1.0  

        if not use_beta_binom:
            rho = 0.0

        mu = k * W_Q
        factor = 1.0 + (k - 1) * rho
        if factor < 1e-9:
            factor = 1e-9
        var = k * W_Q * (1.0 - W_Q) * factor
        return max(0.0, min(1.0, float(_gaussian_tail(mu, var, t))))
    
    def _mix_from_node_hist(self, node, Q_items: Set[int], t: int,
                        extra_cross_pairs: Optional[Dict[Tuple[int,int], float]] = None):
        hist = getattr(node.data, "estimated_jc_hist", None)
        if hist is None:
            return 0.0

        K = len(node.data.this_node_item_set)
        if K <= 0:
            return 0.0

        hist = np.asarray(hist, dtype=np.float64)
        if len(hist) < K + 1:
            hist = np.pad(hist, (0, K + 1 - len(hist)), constant_values=0.0)
        elif len(hist) > K + 1:
            hist = hist[:K + 1]

        alpha_vec = self._alpha_tail_vector(node, Q_items, int(t),
                                            extra_cross_pairs=extra_cross_pairs,
                                            use_beta_binom=True)

        f_hat = float(np.dot(hist, alpha_vec))
        return max(0.0, min(1.0, f_hat))
