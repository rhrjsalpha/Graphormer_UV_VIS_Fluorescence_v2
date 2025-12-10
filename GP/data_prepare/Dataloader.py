import torch
import numpy as np
from torch.utils.data import DataLoader
from ogb.utils.mol import smiles2graph

class GraphDataset:
    def __init__(self, smiles_list, max_nodes=128, multi_hop_max_dist=5):
        """
        A standalone implementation for processing SMILES strings into graph graphormer_data.

        Args:
            smiles_list: List of SMILES strings.
            max_nodes: Maximum number of nodes allowed in a graph (used for padding).
            multi_hop_max_dist: Maximum distance for multi-hop edges.
        """
        self.smiles_list = smiles_list
        self.graphs = [self.validate_graph(smiles2graph(smiles)) for smiles in smiles_list]
        self.max_nodes = max_nodes
        self.multi_hop_max_dist = multi_hop_max_dist

    @staticmethod
    def validate_graph(graph):
        """
        Validate that the graph has the necessary keys and structure.

        Args:
            graph: A graph object returned by smiles2graph.

        Returns:
            The validated graph.

        Raises:
            ValueError: If the graph is missing required keys.
        """
        required_keys = ['num_nodes', 'edge_index', 'edge_feat', 'node_feat']
        for key in required_keys:
            if key not in graph:
                raise ValueError(f"Graph is missing required key: {key}")
        if graph['edge_feat'] is None or len(graph['edge_feat']) == 0:
            raise ValueError("Graph has invalid or missing edge features.")
        return graph

    def preprocess_graph(self, graph):
        """
        Preprocess a single graph to add necessary features and structures.

        Args:
            graph: A graph object returned by smiles2graph.

        Returns:
            A dictionary containing preprocessed graph graphormer_data.
        """
        num_nodes = graph['num_nodes']
        edge_index = graph['edge_index']
        edge_attr = graph.get('edge_feat', None)

        if edge_attr is None:
            raise ValueError("The graph does not contain valid edge features. 'edge_feat' key is missing or None.")

        # Base features
        base_features = torch.tensor(graph['node_feat'], dtype=torch.long) + 1

        # Additional features
        hybridization = torch.randint(0, 3, (num_nodes, 1))  # Mock hybridization feature
        formal_charge = torch.randint(-2, 3, (num_nodes, 1))  # Mock formal charges
        aromaticity = torch.randint(0, 2, (num_nodes, 1))  # Mock aromaticity

        # Combine all features
        node_features = torch.cat([base_features, hybridization, formal_charge, aromaticity], dim=-1)

        # Create adjacency matrix
        adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)
        adj[edge_index[0], edge_index[1]] = True

        # Process edge features
        if len(edge_attr.shape) == 1:
            edge_attr = edge_attr[:, None]
        attn_edge_type = torch.zeros((num_nodes, num_nodes, edge_attr.shape[-1]), dtype=torch.long)
        attn_edge_type[edge_index[0], edge_index[1]] = torch.tensor(edge_attr, dtype=torch.long) + 1

        # Shortest path and multi-hop edge features
        shortest_path = self.compute_shortest_paths(adj.numpy())
        max_dist = min(int(np.amax(shortest_path)), self.multi_hop_max_dist)

        # Edge input for multi-hop connections
        edge_input = self.generate_edge_input(shortest_path, attn_edge_type.numpy(), max_dist)

        return {
            'x': node_features,
            'adj': adj,
            'attn_edge_type': attn_edge_type,
            'shortest_path': torch.tensor(shortest_path, dtype=torch.long),
            'edge_input': torch.tensor(edge_input, dtype=torch.long),
        }

    @staticmethod
    def compute_shortest_paths(adj):
        """
        Compute shortest paths using Floyd-Warshall algorithm.

        Args:
            adj: Adjacency matrix.

        Returns:
            Shortest path distance matrix.
        """
        num_nodes = adj.shape[0]
        dist = np.full((num_nodes, num_nodes), np.inf)
        np.fill_diagonal(dist, 0)

        for i, j in zip(*np.where(adj)):
            dist[i, j] = 1

        for k in range(num_nodes):
            for i in range(num_nodes):
                for j in range(num_nodes):
                    dist[i, j] = min(dist[i, j], dist[i, k] + dist[k, j])

        return dist

    def generate_edge_input(self, shortest_path, attn_edge_type, max_dist):
        """
        Generate multi-hop edge input features.

        Args:
            shortest_path: Shortest path distance matrix.
            attn_edge_type: Edge type tensor.
            max_dist: Maximum distance to consider for multi-hop edges.

        Returns:
            Edge input tensor.
        """
        num_nodes = shortest_path.shape[0]
        edge_input = np.zeros((num_nodes, num_nodes, max_dist, attn_edge_type.shape[-1]), dtype=np.int64)

        for i in range(num_nodes):
            for j in range(num_nodes):
                if shortest_path[i, j] < max_dist:
                    edge_input[i, j, int(shortest_path[i, j]) - 1] = attn_edge_type[i, j]

        return edge_input

    def __getitem__(self, index):
        """Retrieve a preprocessed graph by index."""
        return self.preprocess_graph(self.graphs[index])

    def __len__(self):
        return len(self.graphs)

    def collate(self, batch):
        """
        Collate a batch of graphs into a padded tensor format.

        Args:
            batch: List of preprocessed graph dictionaries.

        Returns:
            A dictionary of padded tensors for the batch.
        """
        max_nodes = min(self.max_nodes, max([b['x'].size(0) for b in batch]))

        x = torch.stack([self.pad_tensor(b['x'], max_nodes) for b in batch])
        adj = torch.stack([self.pad_tensor(b['adj'], max_nodes) for b in batch])
        edge_input = torch.stack([self.pad_tensor(b['edge_input'], max_nodes, pad_dim=3) for b in batch])

        return {'x': x, 'adj': adj, 'edge_input': edge_input}

    @staticmethod
    def pad_tensor(tensor, max_len, pad_dim=2):
        """
        Pad a tensor to the specified length.

        Args:
            tensor: Input tensor.
            max_len: Target length.
            pad_dim: Dimensionality of padding.

        Returns:
            Padded tensor.
        """
        pad_size = [max_len] * pad_dim + list(tensor.shape[pad_dim:])
        print("pad_size",pad_size)
        padded = torch.zeros(pad_size, dtype=tensor.dtype)
        print("padded.shape",padded.shape)
        padded[:tensor.shape[0], :tensor.shape[1]] = tensor
        return padded

# Example Usage
if __name__ == "__main__":
    smiles_list = ["CCO", "CCN", "CCC"]
    dataset = GraphDataset(smiles_list)
    print(dataset[0].keys())
    print("""dataset[0]["x"].size()""", dataset[0]["x"].size())
    print("""dataset[0]["adj"].size()""",dataset[0]["adj"].size())
    print("""dataset[0]["attn_edge_type"].size()""", dataset[0]["attn_edge_type"].size())
    print("""dataset[0]["shortest_path"].size()""", dataset[0]["shortest_path"].size())
    print("""dataset[0]["edge_input"].size()""", dataset[0]["edge_input"].size())

    # Create DataLoader
    dataloader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate)

    for batch in dataloader:
        print(batch)
