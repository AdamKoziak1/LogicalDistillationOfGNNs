from scipy.sparse import coo_matrix

import networkx as nx
from networkx.generators import random_graphs, lattice, small, classic
import numpy as np

import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset

import matplotlib.pyplot as plt

def data(name, kfold, cv_split, seed=0, directed=True):
    rng = np.random.default_rng(seed)
    if name == 'BAMultiShapes' or 'EMLC' in name:
        if name == 'BAMultiShapes':
            datalist = [_generate_BAMultiShapes(rng) for _ in range(8000)] # TODO implement the directed logic for this one
        else:
            datalist = [_generate_EMLC(name, rng, directed) for _ in range(5000)]
        
        n_test = len(datalist) // kfold
        
        train_val_data = datalist[:cv_split * n_test] + datalist[(cv_split + 1) * n_test:]
        train_data, val_data = torch.utils.data.random_split(train_val_data, [len(train_val_data) - n_test, n_test])
        test_data = datalist[cv_split * n_test : (cv_split + 1) * n_test]

        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)

        train_val_batch = Batch.from_data_list(train_val_data)
        test_batch = Batch.from_data_list(test_data)

        return 1, 2, train_loader, val_loader, train_val_batch, test_batch
    else:
        dataset = TUDataset(root='data', name=name)
        n_test = len(dataset) // kfold
        permutation = torch.tensor(rng.permutation(len(dataset)))

        train_val_data = dataset[torch.concat([permutation[:cv_split * n_test], permutation[(cv_split + 1) * n_test:]])]
        train_data, val_data = torch.utils.data.random_split(train_val_data, [len(train_val_data) - n_test, n_test])
        test_data = dataset[permutation[cv_split * n_test : (cv_split + 1) * n_test]]

        train_loader = DataLoader(train_data, batch_size=len(train_data), shuffle=True)
        val_loader = DataLoader(val_data, batch_size=len(val_data), shuffle=False)
        
        train_val_batch = Batch.from_data_list(train_val_data)
        test_batch = Batch.from_data_list(test_data)
        
        return dataset.num_features, dataset.num_classes, train_loader, val_loader, train_val_batch, test_batch
    

def _generate_EMLC(name, rng, directed=True):
    u0 = (rng.random((13, 1)) < 0.5).astype(np.float32)
    u1 = np.ones((13, 1), dtype=np.float32)
    graph = nx.erdos_renyi_graph(13, 0.5, seed=rng.choice(2**32), directed=directed)
    adj = nx.adjacency_matrix(graph).toarray()
    if directed:
        return _generate_EMLC_from_graph_directed(name, u0, u1, adj)
    return _generate_EMLC_from_graph_undirected(name, u0, u1, adj)

def _generate_EMLC_from_graph_directed(name, u0, u1, adj):
    edge_index = _adj_to_edge_index(adj)

    out_adj = adj
    in_adj = adj.T

    degrees_out = out_adj.sum(axis=1)  # Out-degree (sum over columns)
    degrees_in = in_adj.sum(axis=1)   # In-degree (sum over rows)
    degrees = degrees_in + degrees_out

    undirected_adj = _directed_adj_to_undirected(adj)
    match name:
        case 'EMLC0':
            has_more_than_half_u0 = (u0.sum() > 6)
            return Data(x=torch.tensor(u0), edge_index=edge_index, y=int(has_more_than_half_u0))
        case 'EMLC1':
            has_lt_4_or_gt_9_neighbors = (degrees < 4) | (degrees > 9)
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=int(has_lt_4_or_gt_9_neighbors.max()))
        case 'EMLC2':
            has_gt_6_neighbors = degrees > 6
            more_than_half_neighbours_with_gt_6_neighbors = ((undirected_adj @ has_gt_6_neighbors) / degrees.clip(1)) > 0.5
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=(torch.tensor(more_than_half_neighbours_with_gt_6_neighbors).float().mean() > 0.5).long())
        
        case 'EMLC3': # more than half of the nodes have in-degree > 6
            has_gt_2_in_neighbors = degrees_in > 6 # check which nodes have in degree > 6
            y = torch.tensor(has_gt_2_in_neighbors).float().mean() > 0.33 # check if over half of the nodes met the condition
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=y.long())
        
        case 'EMLC4': # more than 3/4 of the nodes have in-degree > 6
            has_gt_2_in_neighbors = degrees_in > 5 # check which nodes have in degree > 6
            y = torch.tensor(has_gt_2_in_neighbors).float().mean() > 0.66 # check if over half of the nodes met the condition
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=y.long())
        
        case 'EMLC5': # more than half of the nodes have out-degree > 6
            has_gt_2_out_neighbors = degrees_out > 5 # check which nodes have out degree > 6
            y = torch.tensor(has_gt_2_out_neighbors).float().mean() > 0.66 # check if over half of the nodes met the condition
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=y.long())
        
        case 'EMLC6': # more than 3/4 of the nodes have out-degree > 6
            has_gt_2_out_neighbors = degrees_out > 5 # check which nodes have out degree > 6
            y = torch.tensor(has_gt_2_out_neighbors).float().mean() > 0.66 # check if over half of the nodes met the condition
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=y.long())
        
        case 'EMLC7': # at least one node has out-degree > 2
            has_gt_2_out_neighbors = degrees_out > 8 # check which nodes have out degree > 2
            y = int(has_gt_2_out_neighbors.max()) # check if one of the nodes met the condition
            #print(y)
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=y)
        
        case 'EMLC8': # at least one node has in-degree > 2
            has_gt_2_in_neighbors = degrees_in > 8 # check which nodes have in degree > 2
            y = int(has_gt_2_in_neighbors.max()) # check if one of the nodes met the condition
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=y)

        case 'EMLC9': # more than half of the nodes have at least half of their in-neighbors with in-degree > 2
            has_gt_2_in_neighbors = degrees_in > 5
            more_than_half_neighbours_with_gt_2_in_neighbors = ((in_adj @ has_gt_2_in_neighbors) / degrees_in.clip(1)) > 0.66
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=(torch.tensor(more_than_half_neighbours_with_gt_2_in_neighbors).float().mean() > 0.5).long())
        
        case 'EMLC10': # more than two of the nodes have at least half of their in-neighbors with in-degree > 2
            has_gt_2_in_neighbors = degrees_in > 6
            more_than_half_neighbours_with_gt_2_in_neighbors = ((in_adj @ has_gt_2_in_neighbors) / degrees_in.clip(1)) > 0.5
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=(torch.tensor(more_than_half_neighbours_with_gt_2_in_neighbors).sum() > 2).long())
        

        # case 'EMLC10': # more than half of the nodes have at least half of their in-neighbors or out-neighbors with in-degree > 2 
        #     has_gt_2_in_neighbors = degrees_in > 2

        #     (((in_adj @ has_gt_2_in_neighbors) | (out_adj @ has_gt_2_in_neighbors)) / degrees.clip(1)) > 0.5


        #     more_than_half_in_neighbours_with_gt_2_in_neighbors = ((in_adj @ has_gt_2_in_neighbors) / degrees.clip(1)) > 0.5
        #     more_than_half_out_neighbours_with_gt_2_in_neighbors = ((out_adj @ has_gt_2_in_neighbors) / degrees.clip(1)) > 0.5
        #     combined = more_than_half_in_neighbours_with_gt_2_in_neighbors or more_than_half_out_neighbours_with_gt_2_in_neighbors
        #     return Data(x=torch.tensor(u1), edge_index=edge_index, y=(torch.tensor(more_than_half_neighbours_with_gt_2_in_neighbors).float().mean() > 0.5).long())

        # TODO add more cases specific to directed stuff (will need to update logic, use the higher parity digraphs and have in/out specific rules)

def _generate_EMLC_from_graph_undirected(name, u0, u1, adj):
    edge_index = _adj_to_edge_index(adj)
    match name:
        case 'EMLC0':
            has_more_than_half_u0 = (u0.sum() > 6)
            return Data(x=torch.tensor(u0), edge_index=edge_index, y=int(has_more_than_half_u0))
        case 'EMLC1':
            has_lt_4_or_gt_9_neighbors = (edge_index[0].bincount() < 4) | (edge_index[0].bincount() > 9)
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=int(has_lt_4_or_gt_9_neighbors.max()))
        case 'EMLC2':
            degrees = adj.sum(1)
            has_gt_6_neighbors = degrees > 6
            more_than_half_neighbours_with_gt_6_neighbors = ((adj @ has_gt_6_neighbors) / degrees.clip(1)) > 0.5
            return Data(x=torch.tensor(u1), edge_index=edge_index, y=(torch.tensor(more_than_half_neighbours_with_gt_6_neighbors).float().mean() > 0.5).long())


# The following code is a modified version of
# https://github.com/steveazzolin/gnn_logic_global_expl/blob/master/datasets/BAMultiShapes/generate_dataset.py
 # TODO maybe add another edge for directed?
def _merge_graphs(g1, g2, nb_random_edges=1, rng=np.random.default_rng(0)):
    mapping = dict()
    max_node = max(g1.nodes())

    i = 1
    for n in g2.nodes():
        mapping[n] = max_node + i
        i = i + 1
    g2 = nx.relabel_nodes(g2,mapping)

    g12 = nx.union(g1,g2)
    for i in range(nb_random_edges):
        e1 = list(g1.nodes())[rng.choice(len(g1.nodes()))]
        e2 = list(g2.nodes())[rng.choice(len(g2.nodes()))]
        g12.add_edge(e1,e2)
    return g12

 # TODO update for directed
def _generate_class1(nb_random_edges, nb_node_ba, rng):
    r = rng.choice(3) 
    
    if r == 0: # W + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-9, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1,g2,nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12,g3,nb_random_edges)
    elif r == 1: # W + H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6-5, 1, seed=rng.choice(2**32))
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1,g2,nb_random_edges, rng)
        g3 = small.house_graph()
        g123 = _merge_graphs(g12,g3,nb_random_edges, rng)
    elif r == 2: # H + G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5-9, 1, seed=rng.choice(2**32))
        g2 = small.house_graph()
        g12 = _merge_graphs(g1,g2,nb_random_edges, rng)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12,g3,nb_random_edges, rng)
    return g123

 # TODO update for directed
def _generate_class0(nb_random_edges, nb_node_ba, rng): 
    r = rng.choice(4)
    
    if r > 3:
        g12 = random_graphs.barabasi_albert_graph(nb_node_ba, 1, seed=rng.choice(2**32)) 
    if r == 0: # W
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-6, 1, seed=rng.choice(2**32)) 
        g2 = classic.wheel_graph(6)
        g12 = _merge_graphs(g1,g2,nb_random_edges)      
    if r == 1: # H
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-5, 1, seed=rng.choice(2**32)) 
        g2 = small.house_graph()
        g12 = _merge_graphs(g1,g2,nb_random_edges)      
    if r == 2: # G
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9, 1, seed=rng.choice(2**32)) 
        g2 = lattice.grid_2d_graph(3, 3)
        g12 = _merge_graphs(g1,g2,nb_random_edges)            
    if r == 3: # All
        g1 = random_graphs.barabasi_albert_graph(nb_node_ba-9-5-6, 1, seed=rng.choice(2**32)) 
        g2 = small.house_graph()
        g12 = _merge_graphs(g1,g2,nb_random_edges)
        g3 = lattice.grid_2d_graph(3, 3)
        g123 = _merge_graphs(g12,g3,nb_random_edges)
        g4 =  classic.wheel_graph(6)
        g12 = _merge_graphs(g123,g4,nb_random_edges)
    return g12

 # TODO update for directed
def _generate_BAMultiShapes(rng):
    nb_node_ba = 40
    r = rng.choice(2)
    
    if r == 0:
        g = _generate_class1(nb_random_edges=1, nb_node_ba=nb_node_ba, rng=rng)
        return Data(x=torch.ones((len(g.nodes()), 1)), edge_index=_adj_to_edge_index(nx.adjacency_matrix(g).toarray()), y=0)
    else:
        g = _generate_class0(nb_random_edges=1, nb_node_ba=nb_node_ba, rng=rng)
        return Data(x=torch.ones((len(g.nodes()), 1)), edge_index=_adj_to_edge_index(nx.adjacency_matrix(g).toarray()), y=1)


def _adj_to_edge_index(adj):
    matrix = coo_matrix(adj)
    return torch.stack([torch.tensor(matrix.row, dtype=torch.long), torch.tensor(matrix.col, dtype=torch.long)])

def EMLC_compare(name):
    rng = np.random.default_rng(0)
    u0 = (rng.random((13, 1)) < 0.5).astype(np.float32)
    u1 = np.ones((13, 1), dtype=np.float32)

    # Generate the undirected graph once
    graph = nx.erdos_renyi_graph(13, 0.5)
    undirected_adj = nx.adjacency_matrix(graph).toarray()

    # Generate undirected Data object
    undirected = _generate_EMLC_from_graph_undirected(name, u0, u1, undirected_adj)

    # Generate directed version of the same graph
    directed_adj = _undirected_adj_to_directed_nodup(undirected_adj)
    directed = _generate_EMLC_from_graph_directed(name, u0, u1, directed_adj)
    print("Y:", directed.y)
    print('adj')
    print(undirected_adj.sum(axis=1), undirected_adj.sum(axis=1) + undirected_adj.sum(axis=0))
    print(undirected_adj)
    print('directed_adj')
    print(directed_adj)
    print(directed_adj.sum(axis=1),directed_adj.sum(axis=0), directed_adj.sum(axis=1) + directed_adj.sum(axis=0))

    print(f"Dataset: {name}")

    # Compare labels
    labels_match = undirected.y == directed.y
    print(f"Undirected label: {undirected.y}")
    print(f"Directed label: {directed.y}")
    print(f"Labels match: {labels_match}\n")

    # Compute degrees for the undirected graph
    num_nodes = undirected.num_nodes
    undirected_edge_index = undirected.edge_index
    undirected_degrees = torch.zeros(num_nodes, dtype=torch.long)
    undirected_degrees.scatter_add_(0, undirected_edge_index[0],
                                    torch.ones(undirected_edge_index.size(1), dtype=torch.long))

    # Compute in-degree, out-degree, and total degree for the directed graph
    directed_edge_index = directed.edge_index

    # In-degree: Count of edges where node is the target
    in_degrees = torch.zeros(num_nodes, dtype=torch.long)
    in_degrees.scatter_add_(0, directed_edge_index[1],
                            torch.ones(directed_edge_index.size(1), dtype=torch.long))

    # Out-degree: Count of edges where node is the source
    out_degrees = torch.zeros(num_nodes, dtype=torch.long)
    out_degrees.scatter_add_(0, directed_edge_index[0],
                             torch.ones(directed_edge_index.size(1), dtype=torch.long))

    # Total degree
    total_degrees = in_degrees + out_degrees

    print("node id, degree (un), degree (di), in, out")
    for node_id in range(num_nodes):
        print(f"{node_id}: {undirected_degrees[node_id]}, {total_degrees[node_id]}, "
              f"In {in_degrees[node_id]}, Out {out_degrees[node_id]}")
    print()

    # Compare adjacency matrices
    undirected_adj = torch.tensor(undirected_adj)
    undir_converted = torch.tensor(_directed_adj_to_undirected(directed_adj))
    adj_match = torch.equal(undir_converted, undirected_adj)
    print(f"Adjacency matrices match: {adj_match}")

    # Optionally, print the adjacency matrices
   # print(f"Undirected adjacency matrix:\n{undirected_adj}")
    #print(f"Directed adjacency matrix:\n{directed_adj_torch}")
    return labels_match

def _undirected_adj_to_directed_nodup(adj, p=0.5):
    # Convert undirected adjacency matrix to directed by assigning a random direction to each edge with some probability, mutually exclusive
    rng = np.random.default_rng(0)
    directed_adj = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1:
                if rng.random() < p:
                    directed_adj[i, j] = 1
                else:
                    directed_adj[j, i] = 1
    return directed_adj

def _undirected_adj_to_directed(adj, p1=0.45, p2=0.9):
    # Convert undirected adjacency matrix to directed by assigning one or both directions as an edge with tunable probabilities
    rng = np.random.default_rng(0)
    directed_adj = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1:
                randval = rng.random()
                if randval < p1:
                    directed_adj[i, j] = 1
                elif randval > p1 and randval < p2:
                    directed_adj[j, i] = 1
                else:
                    directed_adj[i, j] = 1
                    directed_adj[j, i] = 1
    return directed_adj

def _directed_adj_to_undirected(adj):
    undirected_adj = np.zeros_like(adj)
    for i in range(adj.shape[0]):
        for j in range(i + 1, adj.shape[1]):
            if adj[i, j] == 1 or adj[j, i] == 1:
                undirected_adj[i, j] = 1
                undirected_adj[j, i] = 1
    return undirected_adj

def generate_graphs_nodup(num_graphs, num_nodes, edge_prob, p=0.5, seed=0):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        # Generate an undirected graph
        G_undirected = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=rng.choice(2**32))
        # Get adjacency matrix
        adj_undirected = nx.adjacency_matrix(G_undirected).toarray()
        # Convert to directed using nodup method
        adj_directed = _undirected_adj_to_directed_nodup(adj_undirected, p=p)
        # Create directed graph from adjacency matrix
        G_directed = nx.from_numpy_array(adj_directed, create_using=nx.DiGraph)
        graphs.append(G_directed)
    return graphs

def generate_graphs_directed(num_graphs, num_nodes, edge_prob, seed=0):
    rng = np.random.default_rng(seed)
    graphs = []
    for _ in range(num_graphs):
        # Generate a directed graph
        G_directed = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True, seed=rng.choice(2**32))
        graphs.append(G_directed)
    return graphs

def compute_degree_stats(graphs, degree_thresholds):
    in_degree_counts = []
    out_degree_counts = []
    nodes_with_in_degree_threshold = {thresh: [] for thresh in degree_thresholds}
    nodes_with_out_degree_threshold = {thresh: [] for thresh in degree_thresholds}
    
    for G in graphs:
        in_degrees = np.array([deg for _, deg in G.in_degree()])
        out_degrees = np.array([deg for _, deg in G.out_degree()])
        in_degree_counts.extend(in_degrees)
        out_degree_counts.extend(out_degrees)
        for thresh in degree_thresholds:
            num_nodes_in = np.sum(in_degrees >= thresh)
            num_nodes_out = np.sum(out_degrees >= thresh)
            nodes_with_in_degree_threshold[thresh].append(num_nodes_in)
            nodes_with_out_degree_threshold[thresh].append(num_nodes_out)
    
    return in_degree_counts, out_degree_counts, nodes_with_in_degree_threshold, nodes_with_out_degree_threshold

def plot_degree_distributions(in_degrees, out_degrees, method_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(in_degrees, bins=range(max(in_degrees)+2), alpha=0.7, color='skyblue', edgecolor='black')
    plt.title(f'In-Degree Distribution ({method_name})')
    plt.xlabel('In-Degree')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    plt.hist(out_degrees, bins=range(max(out_degrees)+2), alpha=0.7, color='salmon', edgecolor='black')
    plt.title(f'Out-Degree Distribution ({method_name})')
    plt.xlabel('Out-Degree')
    plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()

def plot_threshold_results(degree_thresholds, avg_nodes_in, avg_nodes_out, method_name):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(degree_thresholds, avg_nodes_in, marker='o', linestyle='-', color='blue')
    plt.title(f'Average Nodes with In-Degree ≥ Threshold ({method_name})')
    plt.xlabel('Degree Threshold')
    plt.ylabel('Average Number of Nodes')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(degree_thresholds, avg_nodes_out, marker='o', linestyle='-', color='red')
    plt.title(f'Average Nodes with Out-Degree ≥ Threshold ({method_name})')
    plt.xlabel('Degree Threshold')
    plt.ylabel('Average Number of Nodes')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def graph_degree_stats_test():
    num_graphs = 1000
    num_nodes = 13
    edge_prob = 0.5
    degree_thresholds = list(range(1, num_nodes))
    seed = 42

    # Generate graphs using nodup method
    graphs_nodup = generate_graphs_nodup(num_graphs, num_nodes, edge_prob, p=0.5, seed=seed)
    # Generate graphs directly as directed graphs
    graphs_directed = generate_graphs_directed(num_graphs, num_nodes, edge_prob, seed=seed)

    # Compute degree statistics for nodup graphs
    in_degrees_nodup, out_degrees_nodup, nodes_in_thresh_nodup, nodes_out_thresh_nodup = compute_degree_stats(
        graphs_nodup, degree_thresholds
    )

    # Compute degree statistics for directly generated directed graphs
    in_degrees_directed, out_degrees_directed, nodes_in_thresh_directed, nodes_out_thresh_directed = compute_degree_stats(
        graphs_directed, degree_thresholds
    )

    # Plot degree distributions
    plot_degree_distributions(in_degrees_nodup, out_degrees_nodup, 'Undirected to Directed (nodup)')
    plot_degree_distributions(in_degrees_directed, out_degrees_directed, 'Directly Generated Directed Graphs')

    # Calculate average counts of nodes with degrees above thresholds
    avg_nodes_in_nodup = [np.mean(nodes_in_thresh_nodup[thresh]) for thresh in degree_thresholds]
    avg_nodes_out_nodup = [np.mean(nodes_out_thresh_nodup[thresh]) for thresh in degree_thresholds]

    avg_nodes_in_directed = [np.mean(nodes_in_thresh_directed[thresh]) for thresh in degree_thresholds]
    avg_nodes_out_directed = [np.mean(nodes_out_thresh_directed[thresh]) for thresh in degree_thresholds]

    # Plot threshold results for nodup method
    plot_threshold_results(degree_thresholds, avg_nodes_in_nodup, avg_nodes_out_nodup, 'Undirected to Directed (nodup)')

    # Plot threshold results for directly generated directed graphs
    plot_threshold_results(degree_thresholds, avg_nodes_in_directed, avg_nodes_out_directed, 'Directly Generated Directed Graphs')

    # Optionally, plot both methods on the same plot for comparison
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(degree_thresholds, avg_nodes_in_nodup, marker='o', linestyle='-', color='blue', label='Nodup Method')
    plt.plot(degree_thresholds, avg_nodes_in_directed, marker='x', linestyle='--', color='green', label='Directed Graphs')
    plt.title('Average Nodes with In-Degree ≥ Threshold')
    plt.xlabel('Degree Threshold')
    plt.ylabel('Average Number of Nodes')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(degree_thresholds, avg_nodes_out_nodup, marker='o', linestyle='-', color='red', label='Nodup Method')
    plt.plot(degree_thresholds, avg_nodes_out_directed, marker='x', linestyle='--', color='purple', label='Directed Graphs')
    plt.title('Average Nodes with Out-Degree ≥ Threshold')
    plt.xlabel('Degree Threshold')
    plt.ylabel('Average Number of Nodes')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def check_emlc():
    #names = ['EMLC2']
    names = [#'EMLC3', 
             'EMLC4', 
             #'EMLC5',
             'EMLC6', 
             #'EMLC7', 
             #'EMLC8', 
             'EMLC9', 
             #'EMLC10'
             ]
    #dic = {}
    for name in names:
        #matching = 0
        print(name)
        for i in range(1):
            num_features, num_classes, train_loader, val_loader, train_val_batch, test_batch = data(name, 10, 0, seed=42)    
            bincount = torch.bincount(train_val_batch.y, minlength=2)
            weight = len(train_val_batch) / (2 * bincount.float())
            print(bincount, weight)
            #if EMLC_compare(name):
                #matching +=1 
        #dic[name] = matching
    #print(dic)

if __name__ == "__main__":
    check_emlc()
    #graph_degree_stats_test()

    