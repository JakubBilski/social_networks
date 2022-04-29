import networkx as nx
import numpy as np
from networkx.generators.random_graphs import erdos_renyi_graph, barabasi_albert_graph
import matplotlib.pyplot as plt
from tqdm import tqdm

def simulation(G, beta, gamma, starting_s):
    n = len(G.nodes)
    k = len(G.edges()) / n
    infected = np.zeros(n)
    infected[np.random.choice(np.arange(infected.shape[0]), size=starting_s, replace=False)] = 1
    timestamp = 0
    while True:
        for node in range(n):
            if infected[node] == 0:
                for edge in G.edges(node):
                    if infected[edge[1]] == 1 and np.random.rand() < beta:
                        infected[node] = 1
                        break
        for node in range(n):
            if infected[node] > 0 and np.random.rand() < gamma:
                infected[node] = 0

        # color_map = ['red' if infected[node] == 1.0 else 'blue' for node in range(n)]
        # pos = nx.spring_layout(G, seed=42)
        # plt.clf()
        # np.random.seed(42)
        # nx.draw(G, node_color=color_map, pos=pos, node_size=10, width=0.15)
        # plt.pause(0.1)
        timestamp += 1
        if timestamp == 100:
            return np.sum(infected) / n


if __name__ == '__main__':
    n = 1000
    p = 0.01
    starting_s = 10
    epi_threshold = 1/(p*(n-starting_s))
    print(epi_threshold)
    betas = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
    gammas = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    plt.plot(betas, [b/epi_threshold for b in betas])
    for beta in tqdm(betas):
        for gamma in gammas:
            G = erdos_renyi_graph(n=n, p=p)
            achieved_threshold = beta / gamma
            infected_share = simulation(G, beta, gamma, starting_s)
            plt.scatter([beta], [gamma], c='r' if infected_share > 0.1 else 'g')
    plt.show()
