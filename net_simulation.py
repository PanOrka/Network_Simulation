import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
from random import randint
from statistics import mean


packet_size = 512


def a_func_def(Net, matrix_flow, nodes):
    ''' Obliczanie funkcji przeplywu (wagi krawedzi w grafie podanym w parametrze) '''
    # Pierw czyscimy wagi
    for i, k in Net.edges:
        Net[i][k]['weight'] = 0
    for i in range(nodes-1):
        for k in range(i+1, nodes):
            try:
                path = nx.astar_path(Net, i, k, weight=None)
                for j in range(len(path)-1):
                    Net[path[j]][path[j+1]]['weight'] += matrix_flow[i][k]
                    Net[path[j]][path[j+1]]['weight'] += matrix_flow[k][i]
            except nx.NetworkXNoPath:
                print("Path not found\n")


def c_func_def(Net, nodes, multiplayer=packet_size*15): # mozna wyslac x razy wiecej pakietow
    ''' Obliczanie funkcji przepustowosci (wagi krawedzi w zwroconym grafie) '''
    c_func = Net.copy()
    weights = []
    # liczymy srednia z nie-zerowych wartosci wag, czyli przeplywu
    for i, k in Net.edges:
        if c_func[i][k]['weight'] != 0:
            weights += [c_func[i][k]['weight']]
    avg_weights = mean(weights)
    # mnozymy przez multiplayer wagi lub ustalamy jako srednia z nie-zerowych wartosci wag
    for i, k in Net.edges:
        if c_func[i][k]['weight'] == 0:
            c_func[i][k]['weight'] = multiplayer*int(avg_weights)
        else:
            c_func[i][k]['weight'] = max(c_func[i][k]['weight']*multiplayer, int(avg_weights)*multiplayer) # wyznaczam przepustowosc krawedzi jako max z tych 2 wartosci
    return c_func


def show_graph(graph):
    ''' Rysowanie grafu, odleglosci zachowane jako wagi '''
    pos = nx.kamada_kawai_layout(graph, dist=None)
    nx.draw_networkx(graph, pos)
    labels = nx.get_edge_attributes(graph, 'weight')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)
    plt.show()


def create_network_graph(edges, nodes):
    ''' Generowanie grafu, kazdy wierzcholek ma przynajmniej dwie krawedzie '''
    edges_temp = 0
    Net = nx.Graph()
    for i in range(nodes):
        Net.add_node(i)
    if nodes <= edges:
        rand_perm = np.random.permutation(nodes)
        for i in range(nodes-1):
            edges_temp += 1
            Net.add_edge(rand_perm[i], rand_perm[i+1], weight=0)
        edges_temp += 1
        Net.add_edge(rand_perm[-1], rand_perm[0], weight=0)
    while edges_temp != edges:
        while True:
            i, j = randint(0, nodes-1), randint(0, nodes-1)
            if not Net.has_edge(i, j) and i != j:
                Net.add_edge(i, j, weight=0)
                edges_temp += 1
                break

    return Net


def T(Net, matrix_flow, c_func, m=packet_size):
    matrix_flow_sum = 0
    for row in matrix_flow:
        matrix_flow_sum += sum(row)
    r_sum = 0
    for i, k in Net.edges:
        if ((c_func[i][k]['weight']/m) - Net[i][k]['weight'] > 0):
            r_sum += Net[i][k]['weight']/((c_func[i][k]['weight']/m) - Net[i][k]['weight'])
        else:
            print("FAILURE")
            exit(1)
    return r_sum/matrix_flow_sum


def ts_testing(Net, matrix_flow, c_func, p=90, T_max=0.00115, iterations=1000):
    success = 0
    for i in range(iterations):
        temp_net = Net.copy()
        for i, k in temp_net.edges:
            if p < randint(1, 100) <= 100:
                temp_net.remove_edge(i, k)
                if not nx.is_connected(temp_net): # jesli go rozspojnimy to wracamy
                    temp_net.add_edge(i, k, weight=0)
        a_func_def(temp_net, matrix_flow, len(temp_net))
        actual_T = T(temp_net, matrix_flow, c_func)
        if actual_T < T_max:
            success += 1
    print("Prob =", success/iterations)


def delta_matrix_flow(Net, matrix_flow, c_func, iterations=10):
    temp_matrix_flow = matrix_flow.copy()
    for i in range(iterations):
        for _ in range(randint(5, 2*len(matrix_flow))):
            c1 = randint(0, len(matrix_flow)-1)
            c2 = randint(0, len(matrix_flow)-1)
            while c1 == c2:
                c2 = randint(0, len(matrix_flow)-1)
            delta_packet = randint(1, 10)
            matrix_flow[c1][c2] += delta_packet
        print("////////////////////////////////////////////")
        print("Changed matrix_flow #" + str(i+1))
        print(np.matrix(temp_matrix_flow))
        print("After change #" + str(i+1))
        ts_testing(Net, temp_matrix_flow, c_func)
        print("////////////////////////////////////////////")


def delta_c_func(Net, matrix_flow, c_func, iterations=10):
    temp_c_func = c_func.copy()
    for j in range(iterations):
        for i, k in temp_c_func.edges:
            if randint(0, 1) == 1:
                delta_packet = randint(5, 100)
                temp_c_func[i][k]['weight'] += temp_c_func[i][k]['weight']//delta_packet
        print("////////////////////////////////////////////")
        print("Changed c_func #" + str(j+1))
        ts_testing(Net, matrix_flow, temp_c_func)
        print("////////////////////////////////////////////")


def delta_topology(Net, matrix_flow, c_func, iterations=10):
    avg_weight = 0.0
    for i, k in c_func.edges:
        avg_weight += c_func[i][k]['weight']
    avg_weight /= len(c_func.edges)
    temp_net = Net.copy()
    temp_c_func = c_func.copy()
    for i in range(iterations):
        e1 = randint(0, len(temp_net)-1)
        e2 = randint(0, len(temp_net)-1)
        while e1 == e2 or temp_net.has_edge(e1, e2): # tutaj zmieniam obydwa w razie gdyby nie dalo sie poprowadzic krawedzi z zadnego z nich
            e1 = randint(0, len(temp_net)-1)
            e2 = randint(0, len(temp_net)-1)
        temp_net.add_edge(e1, e2, weight=0)
        temp_c_func.add_edge(e1, e2, weight=avg_weight)
        print("////////////////////////////////////////////")
        print("Changed toplogy #" + str(i+1))
        print("Added edge (" + str(e1) + ", " + str(e2) + ") with c_func avg = " + str(avg_weight))
        ts_testing(temp_net, matrix_flow, temp_c_func)
        print("////////////////////////////////////////////")


def net(edges=29, nodes=20):
    matrix_flow = [[randint(0, 10) if i != k else 0 for k in range(nodes)] for i in range(nodes)] # max 10 pakietow
    print("Starting matrix_flow")
    print(np.matrix(matrix_flow))
    Net = create_network_graph(edges, nodes)
    a_func_def(Net, matrix_flow, nodes) # przeplyw jako wagi grafu
    c_func = c_func_def(Net, nodes) # przepustowosc
    show_graph(Net)
    show_graph(c_func)
    ts_testing(Net, matrix_flow, c_func)
    delta_matrix_flow(Net, matrix_flow, c_func)
    delta_c_func(Net, matrix_flow, c_func)
    delta_topology(Net, matrix_flow, c_func)


if __name__ == "__main__":
    net()