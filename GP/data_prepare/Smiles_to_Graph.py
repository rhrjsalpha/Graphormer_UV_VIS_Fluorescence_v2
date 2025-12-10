from rdkit import Chem
import numpy as np

# 1. Define categorical feature vocabularies
ATOM_FEATURES_VOCAB = {
    'atomic_num': list(range(1, 119)),
    'formal_charge': list(range(-5, 6)),
    'hybridization': [
        Chem.rdchem.HybridizationType.UNSPECIFIED, Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.OTHER
    ],
    'is_aromatic': [0, 1],
    'total_num_hs': list(range(0, 9)),
}

BOND_FEATURES_VOCAB = {
    'bond_type': [
        Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC
    ],
    'stereo': [
        Chem.rdchem.BondStereo.STEREONONE, Chem.rdchem.BondStereo.STEREOANY,
        Chem.rdchem.BondStereo.STEREOZ, Chem.rdchem.BondStereo.STEREOE,
        Chem.rdchem.BondStereo.STEREOCIS, Chem.rdchem.BondStereo.STEREOTRANS
    ],
    'is_conjugated': [0, 1],
    'is_in_ring': [0, 1],
}

def _one_hot_encode(value, vocab):
    if value not in vocab:
        value = vocab[0]
    vec = np.zeros(len(vocab), dtype=np.float32)
    vec[vocab.index(value)] = 1.0
    return vec

def _compute_shortest_paths(adj):
    num_nodes = adj.shape[0]
    dist = np.full((num_nodes, num_nodes), -1, dtype=int)
    np.fill_diagonal(dist, 0)
    for i in range(num_nodes):
        q = [(i, 0)]
        visited = {i}
        head = 0
        while head < len(q):
            u, d = q[head]
            head += 1
            dist[i, u] = d
            for v in np.where(adj[u])[0]:
                if v not in visited:
                    visited.add(v)
                    q.append((v, d + 1))
    dist[dist == -1] = 510
    return dist

def smiles2graph_customized(smiles: str, multi_hop_max_dist: int = 5):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # 혹시 혼합물 있을 경우 배제
    if len(Chem.GetMolFrags(mol)) > 1:
        return None

    num_nodes = mol.GetNumAtoms()
    adj = np.zeros((num_nodes, num_nodes), dtype=bool)

    # Node features
    node_features = {key: [] for key in ATOM_FEATURES_VOCAB}
    for atom in mol.GetAtoms():
        for key, vocab in ATOM_FEATURES_VOCAB.items():
            if key == 'atomic_num':   prop = atom.GetAtomicNum()
            elif key == 'formal_charge': prop = atom.GetFormalCharge()
            elif key == 'hybridization': prop = atom.GetHybridization()
            elif key == 'is_aromatic':   prop = int(atom.GetIsAromatic())
            elif key == 'total_num_hs':  prop = atom.GetTotalNumHs()
            node_features[key].append(_one_hot_encode(prop, vocab))
    for key in node_features:
        node_features[key] = np.array(node_features[key], dtype=np.float32)

    # Edge features
    edge_feat_dim = sum(len(v) for v in BOND_FEATURES_VOCAB.values())
    attn_edge_type = np.zeros((num_nodes, num_nodes, edge_feat_dim), dtype=np.float32)
    edge_indices = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adj[i, j] = adj[j, i] = True
        edge_indices.extend([[i, j], [j, i]])
        
        bond_feats = np.concatenate([
            _one_hot_encode(bond.GetBondType(), BOND_FEATURES_VOCAB['bond_type']),
            _one_hot_encode(bond.GetStereo(), BOND_FEATURES_VOCAB['stereo']),
            _one_hot_encode(int(bond.GetIsConjugated()), BOND_FEATURES_VOCAB['is_conjugated']),
            _one_hot_encode(int(bond.IsInRing()), BOND_FEATURES_VOCAB['is_in_ring'])
        ])
        attn_edge_type[i, j, :] = attn_edge_type[j, i, :] = bond_feats

    spatial_pos = _compute_shortest_paths(adj)
    
    # Generate edge_input from attn_edge_type
    edge_input = np.zeros((num_nodes, num_nodes, multi_hop_max_dist, edge_feat_dim), dtype=np.float32)
    for i in range(num_nodes):
        for j in range(num_nodes):
            dist = spatial_pos[i, j]
            if 1 <= dist < multi_hop_max_dist:
                edge_input[i, j, dist - 1] = attn_edge_type[i, j]

    return {
        'x': node_features,
        'adj': adj,
        'edge_index': np.array(edge_indices).T if edge_indices else np.empty((2, 0), dtype=int),
        'attn_edge_type': attn_edge_type,
        'spatial_pos': spatial_pos,
        'edge_input': edge_input,
        'num_nodes': num_nodes,
    }


# Example Usage
if __name__ == "__main__":
    smiles_list = ["CCO", "CCN", "CCC"]
    for smi in smiles_list:
        feature = smiles2graph_customized(smi)
        #print(feature)
        print(feature.keys())
        print(type(feature["x"]))
        print(type(feature["edge_index"]))