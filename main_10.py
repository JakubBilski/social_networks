import csv
import networkx as nx
from networkx.generators.classic import complete_graph
import numpy as np
import matplotlib.pyplot as plt
from networkx.relabel import convert_node_labels_to_integers

class static_variables:
    NUM_COMMUNITIES = 2

def generate_example_community_graph(num_nodes, num_communities, k_within, k_between):
    adj_matrix = np.zeros((num_nodes, num_nodes))
    community_size = num_nodes // num_communities
    for i in range(num_nodes):
        community = i // community_size
        community_start = community * community_size
        community_end = (community + 1) * community_size
        chosen_neighbors = np.random.choice(list(range(community_start, community_end)), k_within)
        for neigh in chosen_neighbors:
            if neigh > i:
                adj_matrix[i, neigh] = 1
                adj_matrix[neigh, i] = 1
        chosen_neighbors = np.random.choice(list(range(0, community_start)) + list(range(community_end, num_nodes)), k_between)
        for neigh in chosen_neighbors:
            if neigh > i:
                adj_matrix[i, neigh] = 1
                adj_matrix[neigh, i] = 1
    return nx.from_numpy_array(adj_matrix)


def get_best_partition(G, node_list):
    if len(node_list) == 1:
        return np.zeros(len(G.nodes))

    num_nodes = len(G.nodes)
    communities = np.random.randint(0, 2, size=num_nodes)
    kappas = np.zeros(num_nodes)
    aris = np.zeros(2)
    degrees = np.zeros(num_nodes)

    for node in node_list:
        node_community = communities[node]
        for edge in G.edges(node):
            degrees[node] += 1
            if communities[edge[1]] == node_community:
                kappas[node] += 1
                aris[node_community] += 1
            else:
                aris[node_community] += 2

    aris /= 2
    num_edges = np.sum(degrees) / 2
    best_communities = communities.copy()
    global_fitness = np.sum([kappas[i]*num_edges/degrees[i] - aris[communities[i]] for i in node_list])
    best_fitness = global_fitness
    num_tries = 0

    while num_tries < len(node_list):
        num_tries += 1
        worst_nodes = np.argsort([kappas[i]*num_edges/degrees[i] - aris[communities[i]] for i in node_list])
        probabilities = 1/np.power(np.arange(1, worst_nodes.shape[0]+1), 2)
        node = node_list[np.random.choice(worst_nodes, 1, p=probabilities/np.sum(probabilities))[0]]
        # node = node_list[worst_nodes[0]]
        communities[node] = 1-communities[node]
        kappas[node] = degrees[node]-kappas[node]
        for edge in G.edges(node):
            neighbor = edge[1]
            if communities[neighbor] == communities[node]:
                kappas[neighbor] += 1
                aris[1-communities[node]] -= 1
            else:
                kappas[neighbor] -= 1
                aris[communities[node]] += 1

        global_fitness = np.sum([kappas[i]*num_edges/degrees[i] - aris[communities[i]] for i in node_list])

        if global_fitness > best_fitness:
            best_communities = communities.copy()
            best_fitness = global_fitness
            num_tries = 0

    communities = best_communities

    if best_fitness <= 0:
        return np.zeros(len(G.nodes))

    community_0_node_list = [i for i in node_list if communities[i] == 0]
    community_1_node_list = [i for i in node_list if communities[i] == 1]

    if len(community_0_node_list) > 0 and len(community_1_node_list) > 0:

        for edge in list(G.edges):
            if edge[0] in node_list and edge[1] in node_list and\
                    communities[edge[0]] != communities[edge[1]]:
                G.remove_edge(edge[0], edge[1])

        communities_0 = get_best_partition(G, community_0_node_list)
        for i in community_0_node_list:
            communities[i] += communities_0[i]
        
        num_comunities = max(communities_0)
        communities_1 = get_best_partition(G, community_1_node_list)
        for i in community_1_node_list:
            communities[i] += communities_1[i] + num_comunities 

    return communities

if __name__ == '__main__':
    file_name = "D3-UNC"
    adj_matrix = np.genfromtxt(f"./competition_data/{file_name}.csv", delimiter=',')
    G_original = nx.from_numpy_array(adj_matrix)
    # G_original = nx.barbell_graph(20, 3)
    # G_original = generate_example_community_graph(400, 8, 40, 4)
    # G_original = complete_graph(30)
    # G_original = nx.karate_club_graph()
    print(f"Number of nodes: {len(G_original.nodes)}")
    G = convert_node_labels_to_integers(G_original)
    np.random.seed(20)

    communities = get_best_partition(G, list(range(len(G.nodes))))
    unique_communities = sorted(np.unique(communities), key=lambda x:(np.where(communities==x)[0][0]))
    communities = [np.argwhere(unique_communities == c)[0][0]+1 for c in communities]
    print(f"Number of communities: {max(communities)}")
    colors = ["red", "blue", "green", "orange", "grey", "deeppink", "black", "yellow"]
    color_map = [colors[c%8] for c in communities]
    pos = nx.spring_layout(G_original, seed=42)
    np.random.seed(42)
    nx.draw(G_original, node_color=color_map, pos=pos, node_size=10, width=0.15)
    plt.show()

    file = open(f"./competition_results/{file_name}.csv", mode='w')
    for node, community in zip(range(len(communities)), communities):
        file.write(f"{node}, {community}\n")
    file.close()
