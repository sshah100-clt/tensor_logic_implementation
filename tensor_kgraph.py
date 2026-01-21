#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tensor Logic (Domingos-style superposition) on FB15k-237

This script runs TWO clean experiments:

(A) Standard Link Prediction (canonical FB15k-237 splits)
    - Filtered evaluation
    - BOTH tail and head prediction (standard protocol)

(B) 2-hop Compositional Reasoning Benchmark (built from TRAIN only; no test peeking)
    - Extract 2-hop paths from TRAIN
    - Require a UNIQUE direct relation (a, r_direct, c) in TRAIN (strict mode)
    - Remove those direct links ONLY from TRAIN for the composition model to prevent shortcut memorization
    - Evaluate filtered ranking on (a, r_direct, ?) using tensor composition:
          pred = e_a @ R_{r1} @ R_{r2}

Core Tensor Logic construction (DIFFERENTIABLE):
    R_r = E^T A_r E   where A_r is sparse adjacency from TRAIN facts

- Composition benchmark does NOT consult valid/test to select examples.
- Model selection uses validation only, test run once.
"""

import os
import time
import requests
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# =========================
# 1) Setup and Configuration
# =========================

CONFIG = {
    # model/training
    "embedding_dim": 256,
    "batch_size": 1024,
    "learning_rate": 5e-4,
    "epochs": 50,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 13579,

    # evaluation
    "eval_every": 10,
    "train_temperature": 0.1,
    "eval_temperature": 0.1,

    # composition benchmark
    "num_val_paths": 1000,
    "num_test_paths": 1000,
    "seed_paths": 42,

    # performance
    "dl_timeout_sec": 60,
    "data_dir": "./fb15k237_data",
}

def set_seeds(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seeds(CONFIG["seed"])

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("CONFIG:")
for k, v in CONFIG.items():
    print(f"  {k}: {v}")


# =================
# 2) Data Loading
# =================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def download_fb15k237(data_dir: str, timeout: int = 60):
    """
    Downloads FB15k-237 files if missing. Does NOT delete existing files.
    """
    ensure_dir(data_dir)
    base_url = "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/"
    files = ["train.txt", "valid.txt", "test.txt"]

    for f in files:
        out_path = os.path.join(data_dir, f)
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            continue
        print(f"Downloading {f} ...")
        r = requests.get(base_url + f, timeout=timeout)
        r.raise_for_status()
        with open(out_path, "w", encoding="utf-8") as fp:
            fp.write(r.text)

def load_splits(data_dir: str):
    """
    Loads splits and builds entity/relation vocab.
    """
    splits = {}
    entities = set()
    relations = set()
    for split_file in ["train.txt", "valid.txt", "test.txt"]:
        name = split_file.split(".")[0]  # train/valid/test
        path = os.path.join(data_dir, split_file)
        triples = []
        with open(path, "r", encoding="utf-8") as fp:
            for line in fp:
                h, r, t = line.strip().split("\t")
                triples.append((h, r, t))
                entities.add(h); entities.add(t); relations.add(r)
        splits[name] = triples
        print(f"Loaded {name}: {len(triples):,} triples")
    return splits, sorted(list(entities)), sorted(list(relations))

download_fb15k237(CONFIG["data_dir"], timeout=CONFIG["dl_timeout_sec"])
data, entities, relations = load_splits(CONFIG["data_dir"])

print("\nDataset Stats:")
print(f"  Entities: {len(entities):,}")
print(f"  Relations: {len(relations):,}")
print(f"  Train: {len(data['train']):,}  Valid: {len(data['valid']):,}  Test: {len(data['test']):,}")


# ==========================
# 3) Filter Maps (LP standard)
# ==========================

def build_tail_filter_map(triples_list):
    """
    For tail prediction filtering: key (h, r) -> set(tails)
    """
    fm = defaultdict(set)
    for triples in triples_list:
        for h, r, t in triples:
            fm[(h, r)].add(t)
    return fm

def build_head_filter_map(triples_list):
    """
    For head prediction filtering: key (t, r) -> set(heads)
    """
    fm = defaultdict(set)
    for triples in triples_list:
        for h, r, t in triples:
            fm[(t, r)].add(h)
    return fm

# Standard filtered evaluation uses all known true triples for filtering at test-time
all_triples = data["train"] + data["valid"] + data["test"]
filter_tail_all = build_tail_filter_map([all_triples])
filter_head_all = build_head_filter_map([all_triples])

# For validation filtering (common practice): train+valid as the known set
filter_tail_val = build_tail_filter_map([data["train"], data["valid"]])
filter_head_val = build_head_filter_map([data["train"], data["valid"]])

print("\nFilter Maps Built:")
print(f"  Tail filter (VAL known): {len(filter_tail_val):,} keys")
print(f"  Head filter (VAL known): {len(filter_head_val):,} keys")
print(f"  Tail filter (ALL known): {len(filter_tail_all):,} keys")
print(f"  Head filter (ALL known): {len(filter_head_all):,} keys")


# ==========================================================
# 4) Composition Benchmark (TRAIN-ONLY, leakage-safe)
# ==========================================================

def extract_composition_from_train_only(train_triples, num_val=1000, num_test=1000, seed=42):
    """
    Build 2-hop paths (a,r1,b,r2,c) from TRAIN facts only,
    and require a UNIQUE direct TRAIN relation (a, r_direct, c).

    Returns:
        val_paths, test_paths, direct_links_to_remove (triples in TRAIN only)
    where each path tuple is: (a, r1, b, r2, c, r_direct)
    """
    rng = np.random.RandomState(seed)

    # adjacency from TRAIN
    forward = defaultdict(list)
    for h, r, t in train_triples:
        forward[h].append((r, t))

    # direct map from TRAIN only: (a,c)->{rels}
    direct_map = defaultdict(set)
    for a, r, c in train_triples:
        direct_map[(a, c)].add(r)

    nodes = list(forward.keys())
    rng.shuffle(nodes)

    target_total = num_val + num_test
    unique_pairs = set()
    paths = []
    direct_links_to_remove = set()

    for a in nodes:
        if len(paths) >= target_total:
            break
        for r1, b in forward[a]:
            if b not in forward:
                continue
            for r2, c in forward[b]:
                if a == b or b == c or a == c:
                    continue
                if (a, c) in unique_pairs:
                    continue

                # STRICT: require unique direct TRAIN relation a->c
                rels = direct_map.get((a, c), None)
                if not rels or len(rels) != 1:
                    continue

                r_direct = next(iter(rels))
                unique_pairs.add((a, c))
                paths.append((a, r1, b, r2, c, r_direct))
                direct_links_to_remove.add((a, r_direct, c))

                if len(paths) >= target_total:
                    break
            if len(paths) >= target_total:
                break

    rng.shuffle(paths)
    val_paths = paths[:num_val]
    test_paths = paths[num_val:num_val + num_test]

    print("\n[Composition Benchmark - TRAIN ONLY]")
    print(f"  Extracted paths total: {len(paths):,}")
    print(f"  Val paths: {len(val_paths):,}")
    print(f"  Test paths: {len(test_paths):,}")
    print(f"  Direct TRAIN links to remove (for composition training): {len(direct_links_to_remove):,}")

    return val_paths, test_paths, direct_links_to_remove

val_paths, test_paths, links_to_remove_train = extract_composition_from_train_only(
    data["train"],
    num_val=CONFIG["num_val_paths"],
    num_test=CONFIG["num_test_paths"],
    seed=CONFIG["seed_paths"],
)

def summarize_paths(paths, name, topk=10):
    heads = [a for (a, r1, b, r2, c, r_direct) in paths]
    rdir  = [r_direct for (a, r1, b, r2, c, r_direct) in paths]
    ch = Counter(heads); cr = Counter(rdir)
    print("\n" + "-"*70)
    print(f"[SANITY] {name} PATHS SUMMARY")
    print("-"*70)
    print(f"Count: {len(paths)}")
    print(f"Unique heads: {len(ch)}")
    print(f"Unique direct relations: {len(cr)}")
    print(f"Top {topk} heads:")
    for x, cnt in ch.most_common(topk):
        print(f"  {x:<40} {cnt}")
    print(f"Top {topk} direct relations:")
    for x, cnt in cr.most_common(topk):
        print(f"  {x:<40} {cnt}")

summarize_paths(val_paths, "VAL")
summarize_paths(test_paths, "TEST")

# Composition model training uses TRAIN with shortcut links removed
train_comp = [tr for tr in data["train"] if tr not in links_to_remove_train]
print(f"\nComposition TRAIN size (scrubbed train only): {len(train_comp):,} (original {len(data['train']):,})")


# ================================
# 5) Indexing / Dataset / Loader
# ================================

e2i = {e: i for i, e in enumerate(entities)}
r2i = {r: i for i, r in enumerate(relations)}
i2e = {i: e for e, i in e2i.items()}
i2r = {i: r for r, i in r2i.items()}

class KGDataset(Dataset):
    def __init__(self, triples, e2i, r2i):
        self.data = torch.tensor([[e2i[h], r2i[r], e2i[t]] for h, r, t in triples], dtype=torch.long)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

train_lp_ds = KGDataset(data["train"], e2i, r2i)
train_comp_ds = KGDataset(train_comp, e2i, r2i)

train_lp_loader = DataLoader(train_lp_ds, batch_size=CONFIG["batch_size"], shuffle=True)
train_comp_loader = DataLoader(train_comp_ds, batch_size=CONFIG["batch_size"], shuffle=True)


# =========================
# 6) Tensor Logic Model
# =========================

class TensorLogicKG(nn.Module):
    """
    Differentiable Domingos-style superposition:

      R_r = E^T A_r E

    A_r is a sparse adjacency built from TRAIN triples.
    We compute R_r on-the-fly for the relations used in the batch,
    so gradients DO flow through E into the loss.
    """

    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.dim = dim
        self.num_relations = num_relations
        self.entity_emb = nn.Embedding(num_entities, dim)
        nn.init.xavier_uniform_(self.entity_emb.weight)
        self._rel_adj = None  # list of sparse A_r

    def set_relation_triples(self, train_triples_idx, device=None):
        """
        Build sparse adjacency matrices A_r from TRAIN triples (indices).
        train_triples_idx: torch.LongTensor [M,3] columns [h,r,t]
        """
        if device is None:
            device = self.entity_emb.weight.device

        N = self.entity_emb.num_embeddings
        R = self.num_relations

        train_cpu = train_triples_idx.detach().cpu().tolist()
        rel_edges = [[] for _ in range(R)]
        for h, r, t in train_cpu:
            rel_edges[r].append((h, t))

        rel_adj = []
        for r in range(R):
            edges = rel_edges[r]
            if not edges:
                idx = torch.zeros((2, 0), dtype=torch.long)
                val = torch.zeros((0,), dtype=torch.float32)
                A = torch.sparse_coo_tensor(idx, val, (N, N), dtype=torch.float32).coalesce()
                rel_adj.append(A.to(device))
                continue

            idx = torch.tensor(edges, dtype=torch.long).t().contiguous()
            val = torch.ones((idx.shape[1],), dtype=torch.float32)
            A = torch.sparse_coo_tensor(idx, val, (N, N), dtype=torch.float32).coalesce()
            rel_adj.append(A.to(device))

        self._rel_adj = rel_adj

    def _relation_matrix(self, r_idx: int):
        """
        Compute R_r = E^T A_r E (differentiable).
        """
        if self._rel_adj is None:
            raise RuntimeError("Call set_relation_triples(...) first.")

        A = self._rel_adj[r_idx]                     # sparse [N,N]
        E = F.normalize(self.entity_emb.weight, p=2, dim=1)  # [N,d]
        AE = torch.sparse.mm(A, E)                   # [N,d] (backprop to E supported)
        Rr = E.t().matmul(AE)                        # [d,d]
        return Rr

    def get_relation_matrix(self, r_idx_tensor):
        """
        Accepts tensor scalar or 1D batch; returns [d,d] or [B,d,d].
        Computes only needed relations (unique).
        """
        if r_idx_tensor.dim() == 0:
            return self._relation_matrix(int(r_idx_tensor.item()))

        r_cpu = r_idx_tensor.detach().cpu().numpy().tolist()
        uniq = sorted(set(int(x) for x in r_cpu))
        mats = {r: self._relation_matrix(r) for r in uniq}
        out = torch.stack([mats[int(r)] for r in r_cpu], dim=0)  # [B,d,d]
        return out

    def forward(self, h_idx, r_idx):
        """
        Tail direction: pred_t = normalize(e_h @ R_r)
        """
        h = F.normalize(self.entity_emb(h_idx), p=2, dim=1)  # [B,d]
        Rr = self.get_relation_matrix(r_idx)

        if Rr.dim() == 3:
            pred_raw = torch.bmm(h.unsqueeze(1), Rr).squeeze(1)
        else:
            pred_raw = h @ Rr

        return F.normalize(pred_raw, p=2, dim=1)

    def forward_head(self, t_idx, r_idx):
        """
        Head direction: pred_h = normalize(e_t @ R_r^T)
        """
        t = F.normalize(self.entity_emb(t_idx), p=2, dim=1)  # [B,d]
        Rr = self.get_relation_matrix(r_idx)
        if Rr.dim() == 3:
            Rt = Rr.transpose(1, 2)
            pred_raw = torch.bmm(t.unsqueeze(1), Rt).squeeze(1)
        else:
            pred_raw = t @ Rr.t()
        return F.normalize(pred_raw, p=2, dim=1)

    def score_all_tails(self, h_idx, r_idx, temperature=1.0):
        pred = self.forward(h_idx, r_idx)  # [B,d]
        all_emb = F.normalize(self.entity_emb.weight, p=2, dim=1)
        scores = pred @ all_emb.T
        return scores / temperature

    def score_all_heads(self, t_idx, r_idx, temperature=1.0):
        pred = self.forward_head(t_idx, r_idx)  # [B,d]
        all_emb = F.normalize(self.entity_emb.weight, p=2, dim=1)
        scores = pred @ all_emb.T
        return scores / temperature


# =========================
# 7) Evaluation (Filtered, head+tail)
# =========================

def _rank_from_scores(scores_1d: torch.Tensor, target_index: int):
    """
    Compute average rank under ties:
      rank = 1 + #better + (ties-1)/2
    """
    target_score = float(scores_1d[target_index].item())
    better = (scores_1d > target_score).sum().item()
    ties = (scores_1d == target_score).sum().item()
    return 1.0 + better + (ties - 1) / 2.0

def eval_link_prediction_filtered(model, triples, device, filter_tail, filter_head, temperature=0.1, batch_size=256):
    """
    Standard filtered evaluation:
    - Tail prediction for (h,r,t): rank t in scores(h,r,?)
    - Head prediction for (h,r,t): rank h in scores(?,r,t)
    Returns: MRR, H@1,H@3,H@10 plus ranks
    """
    model.eval()
    tail_ranks = []
    head_ranks = []

    with torch.no_grad():
        n_batches = (len(triples) + batch_size - 1) // batch_size
        all_emb = F.normalize(model.entity_emb.weight, p=2, dim=1)

        for bi in range(n_batches):
            start = bi * batch_size
            end = min(start + batch_size, len(triples))
            batch = triples[start:end]

            h_names = [x[0] for x in batch]
            r_names = [x[1] for x in batch]
            t_names = [x[2] for x in batch]

            h_idx = torch.tensor([e2i[h] for h in h_names], device=device)
            r_idx = torch.tensor([r2i[r] for r in r_names], device=device)
            t_idx = torch.tensor([e2i[t] for t in t_names], device=device)

            # Tail scores
            scores_tail = model.score_all_tails(h_idx, r_idx, temperature=temperature)  # [B,N]
            # Head scores
            scores_head = model.score_all_heads(t_idx, r_idx, temperature=temperature)  # [B,N]

            for j in range(len(batch)):
                h, r, t = h_names[j], r_names[j], t_names[j]
                h_i = e2i[h]
                t_i = e2i[t]

                # ---- Tail filtering for (h,r,?) ----
                st = scores_tail[j].clone()
                key_t = (h, r)
                if key_t in filter_tail:
                    for known_t in filter_tail[key_t]:
                        if known_t != t:
                            st[e2i[known_t]] = -1e9
                tail_ranks.append(_rank_from_scores(st, t_i))

                # ---- Head filtering for (?,r,t) ----
                sh = scores_head[j].clone()
                key_h = (t, r)
                if key_h in filter_head:
                    for known_h in filter_head[key_h]:
                        if known_h != h:
                            sh[e2i[known_h]] = -1e9
                head_ranks.append(_rank_from_scores(sh, h_i))

    ranks = tail_ranks + head_ranks
    mrr = float(np.mean([1.0 / r for r in ranks]))
    h1 = float(np.mean([1.0 if r <= 1 else 0.0 for r in ranks]))
    h3 = float(np.mean([1.0 if r <= 3 else 0.0 for r in ranks]))
    h10 = float(np.mean([1.0 if r <= 10 else 0.0 for r in ranks]))
    return mrr, h1, h3, h10, ranks

def eval_composition_filtered(model, paths, device, filter_tail, temperature=0.1):
    """
    For each (a,r1,b,r2,c,r_direct):
      pred = e_a @ R_{r1} @ R_{r2}
      rank c among scores for query (a, r_direct, ?)
    Filtering is on (a, r_direct) -> true tails (standard filtered).
    """
    model.eval()
    ranks = []

    with torch.no_grad():
        all_emb = F.normalize(model.entity_emb.weight, p=2, dim=1)

        for (a, r1, b, r2, c, r_direct) in paths:
            a_i = e2i[a]
            c_i = e2i[c]
            r1_i = r2i[r1]
            r2_i = r2i[r2]

            a_idx = torch.tensor([a_i], device=device)
            e_a = F.normalize(model.entity_emb(a_idx), p=2, dim=1)  # [1,d]

            M1 = model.get_relation_matrix(torch.tensor(r1_i, device=device))  # [d,d]
            M2 = model.get_relation_matrix(torch.tensor(r2_i, device=device))  # [d,d]

            pred = F.normalize(e_a @ M1 @ M2, p=2, dim=1)           # [1,d]
            scores = (pred @ all_emb.T).squeeze(0) / temperature     # [N]

            # Filter on (a, r_direct)
            key = (a, r_direct)
            if key in filter_tail:
                for known_t in filter_tail[key]:
                    if known_t != c:
                        scores[e2i[known_t]] = -1e9

            ranks.append(_rank_from_scores(scores, c_i))

    mrr = float(np.mean([1.0 / r for r in ranks])) if ranks else 0.0
    h1 = float(np.mean([1.0 if r <= 1 else 0.0 for r in ranks])) if ranks else 0.0
    h3 = float(np.mean([1.0 if r <= 3 else 0.0 for r in ranks])) if ranks else 0.0
    h10 = float(np.mean([1.0 if r <= 10 else 0.0 for r in ranks])) if ranks else 0.0
    return mrr, h1, h3, h10, ranks


# =========================
# 8) Training Loop Helper
# =========================

def train_model(
    model,
    train_loader,
    device,
    epochs,
    eval_every,
    eval_fn,
    eval_args,
    ckpt_path,
    lr,
    weight_decay=1e-5,
    train_temperature=0.1,
):
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_val = -1.0
    losses = []
    total_pure_training_time = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        t0 = time.time()

        for batch in train_loader:
            batch = batch.to(device)
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]

            optimizer.zero_grad()
            # Tail prediction loss
            scores_tail = model.score_all_tails(h, r, temperature=train_temperature)
            loss_tail = F.cross_entropy(scores_tail, t)
            # Head prediction loss
            scores_head = model.score_all_heads(t, r, temperature=train_temperature)
            loss_head = F.cross_entropy(scores_head, h)
            # Combined loss
            loss = (loss_tail + loss_head) / 2.0
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss.item())
        
        total_pure_training_time += (time.time() - t0)
        avg_loss = total_loss / max(1, len(train_loader))
        losses.append(avg_loss)

        if epoch % 10 == 0:
            print(f"Epoch {epoch:3d}/{epochs}: Loss={avg_loss:.4f}")

        if epoch % eval_every == 0:
            print(f"\n{'='*70}\nVALIDATION @ Epoch {epoch}\n{'='*70}")
            val_mrr, val_h1, val_h3, val_h10, _ = eval_fn(model, **eval_args)
            print(f"VAL: MRR={val_mrr:.4f} H@1={val_h1:.4f} H@3={val_h3:.4f} H@10={val_h10:.4f}")

            if val_mrr > best_val:
                best_val = val_mrr
                torch.save(model.state_dict(), ckpt_path)
                print(f"  ✓ Saved best checkpoint: {ckpt_path}\n")
            else:
                print()

    return best_val, losses, total_pure_training_time


# ==========================================================
# 9) EXPERIMENT A: Standard Link Prediction (canonical splits)
# ==========================================================

print("\n" + "="*80)
print("EXPERIMENT A: STANDARD LINK PREDICTION (canonical FB15k-237)")
print("="*80)

model_lp = TensorLogicKG(len(entities), len(relations), CONFIG["embedding_dim"]).to(CONFIG["device"])
model_lp.set_relation_triples(train_lp_ds.data, device=CONFIG["device"])

best_lp_val, lp_losses, train_time_a = train_model(
    model=model_lp,
    train_loader=train_lp_loader,
    device=CONFIG["device"],
    epochs=CONFIG["epochs"],
    eval_every=CONFIG["eval_every"],
    eval_fn=eval_link_prediction_filtered,
    eval_args=dict(
        triples=data["valid"],
        device=CONFIG["device"],
        filter_tail=filter_tail_val,
        filter_head=filter_head_val,
        temperature=CONFIG["eval_temperature"],
        batch_size=256,
    ),
    ckpt_path="best_model_lp.pth",
    lr=CONFIG["learning_rate"],
    train_temperature=CONFIG["train_temperature"],
)

print("\nLoading best LP checkpoint and testing once...")
model_lp.load_state_dict(torch.load("best_model_lp.pth", map_location=CONFIG["device"]))

lp_mrr, lp_h1, lp_h3, lp_h10, lp_ranks = eval_link_prediction_filtered(
    model_lp,
    triples=data["test"],
    device=CONFIG["device"],
    filter_tail=filter_tail_all,
    filter_head=filter_head_all,
    temperature=CONFIG["eval_temperature"],
    batch_size=256,
)

print("\n[LINK PREDICTION TEST RESULTS - canonical splits]")
print(f"MRR={lp_mrr:.4f}  H@1={lp_h1:.4f}  H@3={lp_h3:.4f}  H@10={lp_h10:.4f}")
print(f"Pure Training Time: {train_time_a:.2f}s")

# ==========================================================
# 10) EXPERIMENT B: Composition (TRAIN-only benchmark, scrubbed TRAIN)
# ==========================================================

print("\n" + "="*80)
print("EXPERIMENT B: 2-HOP COMPOSITION (benchmark built from TRAIN only; scrubbed TRAIN)")
print("="*80)

model_comp = TensorLogicKG(len(entities), len(relations), CONFIG["embedding_dim"]).to(CONFIG["device"])
model_comp.set_relation_triples(train_comp_ds.data, device=CONFIG["device"])

best_comp_val, comp_losses, train_time_b = train_model(
    model=model_comp,
    train_loader=train_comp_loader,
    device=CONFIG["device"],
    epochs=CONFIG["epochs"],
    eval_every=CONFIG["eval_every"],
    eval_fn=eval_composition_filtered,
    eval_args=dict(
        paths=val_paths,
        device=CONFIG["device"],
        filter_tail=filter_tail_val,  # Use train+valid only during validation
        temperature=CONFIG["eval_temperature"],
    ),
    ckpt_path="best_model_comp.pth",
    lr=CONFIG["learning_rate"],
    train_temperature=CONFIG["train_temperature"],
)

print("\nLoading best COMPOSITION checkpoint and testing once...")
model_comp.load_state_dict(torch.load("best_model_comp.pth", map_location=CONFIG["device"]))

comp_mrr, comp_h1, comp_h3, comp_h10, comp_ranks = eval_composition_filtered(
    model_comp,
    paths=test_paths,
    device=CONFIG["device"],
    filter_tail=filter_tail_all,
    temperature=CONFIG["eval_temperature"],
)

print("\n[COMPOSITION TEST RESULTS - leakage-safe benchmark]")
print(f"MRR={comp_mrr:.4f}  H@1={comp_h1:.4f}  H@3={comp_h3:.4f}  H@10={comp_h10:.4f}")
print(f"Pure Training Time: {train_time_b:.2f}s")

# =========================
# 11) Summary
# =========================

print("\n" + "="*80)
print("EXPERIMENT SUMMARY")
print("="*80)
print("A) Link Prediction (canonical splits, head+tail filtered):")
print(f"   Best VAL MRR seeing train+valid truths: {best_lp_val:.4f}")
print(f"   TEST MRR/Hits: MRR={lp_mrr:.4f} H@1={lp_h1:.4f} H@3={lp_h3:.4f} H@10={lp_h10:.4f}")
print(f"   Pure Training Time: {train_time_a:.2f}s")

print("\nB) Composition (TRAIN-only benchmark; scrubbed TRAIN for no-shortcut training):")
print(f"   Best VAL MRR: {best_comp_val:.4f}")
print(f"   TEST MRR/Hits: MRR={comp_mrr:.4f} H@1={comp_h1:.4f} H@3={comp_h3:.4f} H@10={comp_h10:.4f}")
print(f"   Pure Training Time: {train_time_b:.2f}s")

print("\nCleanliness Notes:")
print(" - Composition benchmark construction uses TRAIN only (no valid/test peeking).")
print(" - Tensor Logic superposition R_r = E^T A_r E is differentiable (grad flows through E).")
print(" - Model selection uses validation only; test evaluated once per experiment.")
print("="*80)
