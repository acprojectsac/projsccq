import math, random, hashlib, os
import numpy as np
from typing import Iterable, List, Set, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp
from math import comb

_GLOBAL_HASHER = None

class _MinHasher:
    def __init__(
        self,
        L: int,
        domain_size: int,
        seed: int = 2025,
        output_domain_size: Optional[int] = None
    ):
        assert L > 0 and domain_size > 0
        self.L = L
        self.D = domain_size
        self.seed = int(seed)
        self.outD = None if output_domain_size is None else int(output_domain_size)
        if self.outD is not None:
            assert self.outD >= 2, "output_domain_size 2"

        self.perm_ranks: List[List[int]] = []
        for ell in range(L):
            mix = ((self.seed << 32) ^ (ell * 0x9E3779B97F4A7C15)) & ((1 << 64) - 1)
            rng = random.Random(mix)
            perm = list(range(self.D))
            rng.shuffle(perm)
            rank = [0] * self.D
            for pos, elem in enumerate(perm):
                rank[elem] = pos
            self.perm_ranks.append(rank)

    def _map_to_out(self, ell: int, x: int) -> int:
        if self.outD is None:
            return x
        key = (((self.seed & ((1<<64)-1)) ^ (ell * 0xD6E8FEB86659FD93)) & ((1<<64)-1)).to_bytes(8, 'big')
        h = hashlib.blake2b(x.to_bytes(8, 'big', signed=False), digest_size=8, key=key).digest()
        hv = int.from_bytes(h, 'big')
        return hv % self.outD

    def signature_ids(self, items: Iterable[int]) -> List[int]:
        sig_ids = [-1] * self.L
        items = [x for x in items if 0 <= x < self.D]
        if not items:
            return sig_ids

        for ell in range(self.L):
            rank = self.perm_ranks[ell]
            best_pos = self.D + 1
            best_elem = -1
            for x in items:
                rx = rank[x]
                if rx < best_pos:
                    best_pos = rx
                    best_elem = x
            sig_ids[ell] = self._map_to_out(ell, best_elem) if best_elem >= 0 else -1

        return sig_ids
def _init_global_hasher_for_fork():
    return

def _init_global_hasher_for_spawn(L, D, seed, outD):
    global _GLOBAL_HASHER
    _GLOBAL_HASHER = _MinHasher(L=L, domain_size=D, seed=seed, output_domain_size=outD)

def _set_global_hasher_ref_in_parent(hasher):
    global _GLOBAL_HASHER
    _GLOBAL_HASHER = hasher

def _worker_sig_ids(items):
    return _GLOBAL_HASHER.signature_ids(items)

def compute_user_sigs_process(hasher, user_sets, max_workers=8, chunksize=64):
    n = len(user_sets)
    results = [None] * n

    _set_global_hasher_ref_in_parent(hasher)

    try:
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_global_hasher_for_fork
        ) as ex:
            for i, sig in enumerate(ex.map(_worker_sig_ids, user_sets, chunksize=chunksize)):
                results[i] = sig
        return results
    except Exception:
        L = hasher.L
        D = hasher.D
        seed = hasher.seed
        outD = getattr(hasher, "outD", None)

        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_global_hasher_for_spawn,
            initargs=(L, D, seed, outD)
        ) as ex:
            for i, sig in enumerate(ex.map(_worker_sig_ids, user_sets, chunksize=chunksize)):
                results[i] = sig
        return results

def compute_query_sigs_process(hasher, queries, max_workers=8, chunksize=8):
    n = len(queries)
    results = [None] * n

    _set_global_hasher_ref_in_parent(hasher)

    try:
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_global_hasher_for_fork
        ) as ex:
            for i, sig in enumerate(ex.map(_worker_sig_ids, queries, chunksize=chunksize)):
                results[i] = sig
        return results
    except Exception:
        L = hasher.L
        D = hasher.D
        seed = hasher.seed
        outD = getattr(hasher, "outD", None)

        ctx = mp.get_context('spawn')
        with ProcessPoolExecutor(
            max_workers=max_workers,
            mp_context=ctx,
            initializer=_init_global_hasher_for_spawn,
            initargs=(L, D, seed, outD)
        ) as ex:
            for i, sig in enumerate(ex.map(_worker_sig_ids, queries, chunksize=chunksize)):
                results[i] = sig
        return results