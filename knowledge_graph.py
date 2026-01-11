import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import requests
from collections import defaultdict

CONFIG = {
    'embedding_dim': 256,
    'batch_size': 2048,
    'learning_rate': 0.0005,
    'epochs': 150,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'eval_every': 25,
    'num_test_paths': 500
}

class TensorLogicKG(nn.Module):
    def __init__(self, num_entities, num_relations, dim):
        super().__init__()
        self.dim = dim
        self.entity_emb = nn.Embedding(num_entities, dim)
        self.relation_mat = nn.Embedding(num_relations, dim * dim)
        
        nn.init.xavier_uniform_(self.entity_emb.weight)
        identity = torch.eye(dim).flatten()
        with torch.no_grad():
            self.relation_mat.weight.data.copy_(identity.repeat(num_relations, 1))
            self.relation_mat.weight.data += torch.randn_like(self.relation_mat.weight) * 0.001
    
    def forward(self, h_idx, r_idx):
        h = F.normalize(self.entity_emb(h_idx), p=2, dim=1)
        r = self.relation_mat(r_idx).view(-1, self.dim, self.dim)
        return torch.einsum('bi,bij->bj', h, r)
    
    def score_all(self, h_idx, r_idx):
        pred = self(h_idx, r_idx)
        all_emb = F.normalize(self.entity_emb.weight, p=2, dim=1)
        return torch.matmul(pred, all_emb.T)

def load_data():
    # Delete old files
    for f in ['train.txt', 'valid.txt', 'test.txt']:
        if os.path.exists(f):
            os.remove(f)
    
    url = "https://raw.githubusercontent.com/DeepGraphLearning/KnowledgeGraphEmbedding/master/data/FB15k-237/"
    files = ['train.txt', 'valid.txt', 'test.txt']
    data = {}
    entities, relations = set(), set()
    
    for f in files:
        print(f"Downloading {f}...")
        r = requests.get(url + f)
        with open(f, 'w') as file:
            file.write(r.text)
        
        triples = []
        with open(f, 'r') as file:
            for line in file:
                parts = line.strip().split('\t')
                if len(parts) == 3:
                    h, r, t = parts
                    triples.append((h, r, t))
                    entities.add(h)
                    entities.add(t)
                    relations.add(r)
        
        data[f.split('.')[0]] = triples
        print(f"  Loaded {len(triples)} triples")
    
    return data, sorted(list(entities)), sorted(list(relations))

def extract_compositional_test_set(train_triples, num_paths=500):
    print("\nExtracting compositional paths...")
    
    forward = defaultdict(list)
    all_links = {}
    
    for h, r, t in train_triples:
        forward[h].append((r, t))
        all_links[(h, t)] = r
    
    test_paths = []
    direct_links = set()
    
    entities = list(forward.keys())
    np.random.shuffle(entities)
    
    for a in entities:
        if len(test_paths) >= num_paths:
            break
            
        for r1, b in forward[a]:
            if b not in forward:
                continue
                
            for r2, c in forward[b]:
                if a == c or a == b or b == c:
                    continue
                
                if (a, c) in all_links:
                    r_direct = all_links[(a, c)]
                    test_paths.append((a, r1, b, r2, c))
                    direct_links.add((a, r_direct, c))
                    
                    if len(test_paths) >= num_paths:
                        break
    
    filtered_train = [t for t in train_triples if t not in direct_links]
    
    print(f"Found {len(test_paths)} paths, removed {len(direct_links)} direct links")
    print(f"Training: {len(train_triples)} → {len(filtered_train)}")
    
    return test_paths, direct_links, filtered_train

def test_composition(model, test_paths, e2i, r2i, device):
    model.eval()
    ranks = []
    
    with torch.no_grad():
        for a, r1, b, r2, c in test_paths:
            a_idx = torch.tensor([e2i[a]], device=device)
            r1_idx = torch.tensor([r2i[r1]], device=device)
            r2_idx = torch.tensor([r2i[r2]], device=device)
            c_idx = e2i[c]
            
            emb_a = F.normalize(model.entity_emb(a_idx), p=2, dim=1)
            M1 = model.relation_mat(r1_idx).view(model.dim, model.dim)
            M2 = model.relation_mat(r2_idx).view(model.dim, model.dim)
            
            pred = F.normalize(emb_a @ M1 @ M2, p=2, dim=1)
            
            all_emb = F.normalize(model.entity_emb.weight, p=2, dim=1)
            scores = (pred @ all_emb.T).squeeze()
            
            rank = 1 + (scores > scores[c_idx]).sum().item()
            ranks.append(rank)
    
    mrr = np.mean([1.0 / r for r in ranks])
    h10 = sum(1 for r in ranks if r <= 10) / len(ranks)
    
    print(f"\nComposition Test: MRR={mrr:.4f}, Hits@10={h10:.4f}")
    return mrr

class KGDataset(Dataset):
    def __init__(self, triples, e2i, r2i):
        self.data = torch.tensor([[e2i[h], r2i[r], e2i[t]] for h, r, t in triples], dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def main():
    print("TENSOR LOGIC COMPOSITIONAL REASONING\n")
    
    data, entities, relations = load_data()
    e2i = {e: i for i, e in enumerate(entities)}
    r2i = {r: i for i, r in enumerate(relations)}
    
    print(f"\nDataset: {len(entities)} entities, {len(relations)} relations")
    
    test_paths, _, filtered_train = extract_compositional_test_set(data['train'], CONFIG['num_test_paths'])
    
    train_dataset = KGDataset(filtered_train, e2i, r2i)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    model = TensorLogicKG(len(entities), len(relations), CONFIG['embedding_dim']).to(CONFIG['device'])
    optimizer = optim.AdamW(model.parameters(), lr=CONFIG['learning_rate'])
    
    print("\nTraining...")
    best = 0
    
    for epoch in range(1, CONFIG['epochs'] + 1):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            batch = batch.to(CONFIG['device'])
            h, r, t = batch[:, 0], batch[:, 1], batch[:, 2]
            
            optimizer.zero_grad()
            loss = F.cross_entropy(model.score_all(h, r) / 0.1, t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss={total_loss/len(train_loader):.4f}")
        
        if epoch % CONFIG['eval_every'] == 0:
            mrr = test_composition(model, test_paths, e2i, r2i, CONFIG['device'])
            if mrr > best:
                best = mrr
                print(f"New best: {mrr:.4f}")
    
    print(f"\nDone. Best MRR: {best:.4f}")

if __name__ == "__main__":
    main()