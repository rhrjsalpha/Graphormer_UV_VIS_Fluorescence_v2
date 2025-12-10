import numpy as np

# --- BFS & path reconstruction ---
def _bfs_parents(adj: np.ndarray, src: int):
    n = adj.shape[0]
    prev = -np.ones(n, dtype=np.int32)
    dist = np.full(n, -1, dtype=np.int32)
    q = [src]
    dist[src] = 0
    head = 0
    while head < len(q):
        u = q[head]; head += 1
        for v in np.where(adj[u])[0]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                prev[v] = u
                q.append(v)
    return prev, dist

def _reconstruct_path(prev: np.ndarray, src: int, dst: int):
    if src == dst or prev[dst] == -1:
        return []
    path = [dst]
    cur = dst
    while cur != src and cur != -1:
        cur = prev[cur]
        path.append(cur)
    if cur != src:
        return []
    path.reverse()
    return path

# ===================== 예시 그래프 정의 =====================
# 0-1-2-3-0 사각형 + 대각 1-3 (조금 더 다양한 경로)
adj = np.array([
    [0,1,0,1,0],
    [1,0,1,1,0],
    [0,1,0,1,1],
    [1,1,1,0,0],
    [0,0,1,0,0],
], dtype=np.int32)
num_nodes = adj.shape[0]

# 엣지 타입 예시(원-핫, dim=4): SINGLE, DOUBLE, AROMATIC, OTHER 같은 느낌으로 가정
feat_dim = 4
attn_edge_type = np.zeros((num_nodes, num_nodes, feat_dim), dtype=np.int64)

# 간단한 규칙으로 타입 부여(예시):
# (i+j)%4 로 타입 결정
for i in range(num_nodes):
    for j in range(num_nodes):
        if adj[i, j] == 1:
            t = (i + j) % feat_dim
            attn_edge_type[i, j, t] = 1

multi_hop_max_dist = 5
edge_input = np.zeros((num_nodes, num_nodes, multi_hop_max_dist, feat_dim), dtype=np.int64)

# ===================== edge_input 채우기 =====================
for i in range(num_nodes):
    prev, dist_vec = _bfs_parents(adj, i)
    for j in range(num_nodes):
        d = dist_vec[j]
        if 1 <= d < multi_hop_max_dist:
            path = _reconstruct_path(prev, i, j)  # [i, ..., j]
            if len(path) != d + 1:
                continue
            for h in range(d):  # hop index: 0..d-1
                u, v = path[h], path[h+1]
                edge_input[i, j, h, :] = attn_edge_type[u, v, :]

# ===================== 출력(요약 + 레이어별) =====================
print("edge_input shape: (src, dst, hop, feat) =", edge_input.shape)

# 레이어(hop)별 존재 유무(합계) 매트릭스와 nonzero 합 출력
for h in range(multi_hop_max_dist):
    layer_mat = edge_input[:, :, h, :].sum(axis=-1)  # [N,N]
    nnz = np.count_nonzero(layer_mat)
    if nnz == 0:
        # 완전히 비어있는 레이어는 스킵해도 되고, 보기 위해 출력해도 됨
        print(f"\n[HOP {h+1}] (all zeros)")
        print(layer_mat)
    else:
        print(f"\n[HOP {h+1}] nonzero entries = {nnz}")
        print(layer_mat)

# ===================== 임의의 (src,dst) 케이스 상세 보기 =====================
def print_case(src, dst):
    # 경로/거리/각 hop별 one-hot
    prev, dist_vec = _bfs_parents(adj, src)
    d = dist_vec[dst]
    print(f"\nCase src={src} -> dst={dst} | dist={d}")
    path = _reconstruct_path(prev, src, dst)
    if not path:
        print("  (no path)")
        return
    print("  path:", " -> ".join(map(str, path)))
    for h in range(min(d, multi_hop_max_dist)):
        oh = edge_input[src, dst, h, :]
        if oh.sum() == 0:
            break
        print(f"  hop {h+1} one-hot:", oh.tolist())

# 몇 케이스 예시 출력
print_case(0, 2)  # 다중 hop 경로 예시
print_case(0, 3)  # 1-hop 가능
print_case(1, 4)  # 2~3-hop 예시

# === (추가) 예시 그래프(adj) 시각화 ===
import networkx as nx
import matplotlib.pyplot as plt

def draw_adj_graph(adj: np.ndarray, node_labels=None, layout="circular", title="Example Graph (with node indices)"):
    """
    adj: [N,N] 0/1 or bool adjacency
    node_labels: None 이면 0..N-1, 아니면 길이 N의 문자열/정수 리스트
    layout: "circular" | "spring"
    """
    N = adj.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(N))
    for i in range(N):
        for j in range(i+1, N):
            if adj[i, j]:
                G.add_edge(i, j)

    if node_labels is None:
        labels = {i: str(i) for i in range(N)}
    else:
        labels = {i: str(node_labels[i]) for i in range(N)}

    if layout == "spring":
        pos = nx.spring_layout(G, seed=42)
    else:
        # circular default
        pos = nx.circular_layout(G)

    plt.figure(figsize=(6, 6))
    nx.draw_networkx_nodes(G, pos, node_size=800)
    nx.draw_networkx_edges(G, pos, width=2)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight="bold")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# 호출 예시: 위에서 정의한 adj를 그대로 그림
draw_adj_graph(adj, layout="circular", title="Example Graph from 'adj' (nodes numbered)")
