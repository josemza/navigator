import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import timeit
from functools import partial

class Node:
    def __init__(self, id, goal, deadend, heuristic):
        self.id = id
        self.goal = goal
        self.deadend = deadend
        self.heuristic = heuristic
        self.adjacencies = []  # List of adjacencies

    def add_adjacency(self, node_name, actions):
        self.adjacencies.append((node_name, actions))

    def __repr__(self):
        return f"Node({self.id}, Goal={self.goal}, DeadEnd={self.deadend}, Heuristic={self.heuristic}, Adjacencies={self.adjacencies})"

class Own_graph:
    def __init__(self, json_file):
        self.rows = int(json_file[json_file.find('navigator') + 9]) if 'navigator' in json_file else 5  # default to 5 columns if not specified
        self.json_file = json_file
        self.nodes = {}
        self.G = nx.MultiDiGraph()
        self.goal = str
        self.deadend = str
        self.load_nodes_from_json()
        self.create_graph_from_nodes()
        self.columns = (len(self.G.nodes())) // self.rows
        self.start = str(self.rows * (self.columns - 1) + 1)

    def load_nodes_from_json(self):
        with open(self.json_file, 'r') as file:
            data = json.load(file)

        #for key, value in data.items():
        for i in range(len(data)):
            key = str(i+1)
            value = data[key]
            
            if value['goal']:
                self.goal = key
            
            if value['deadend']:
                self.deadend = key

            node = Node(key, value['goal'], value['deadend'], value['heuristic'])
            for adj in value['Adj']:
                node.add_adjacency(adj['name'], adj['A'])
            self.nodes[key] = node

    def create_graph_from_nodes(self):
        # Añadir nodos
        for node_id, node in self.nodes.items():
            self.G.add_node(node_id, goal=node.goal, deadend=node.deadend, heuristic=node.heuristic)

        # Añadir aristas con pesos y probabilidades
        for node_id, node in self.nodes.items():
            for adj_name, actions in node.adjacencies:
                for action, prob in actions.items():
                    self.G.add_edge(node_id, adj_name, action=action, weight=1, probability=prob)

    def visualize_node_and_neighbors(self, node_id):
        if node_id in self.G:
            # Crear un subgrafo con el nodo y sus vecinos
            neighbors = list(self.G.neighbors(node_id))
            subgraph_nodes = neighbors + [node_id]
            subG = self.G.subgraph(subgraph_nodes)

            pos = nx.spring_layout(subG)  # Posiciones para un buen layout
            nx.draw(subG, pos, with_labels=True, node_color='lightblue', node_size=3000)
            labels = nx.get_edge_attributes(subG, 'probability')
            nx.draw_networkx_edge_labels(subG, pos, edge_labels=labels)

            # Destacar el nodo principal
            nx.draw_networkx_nodes(subG, pos, nodelist=[node_id], node_color='red', node_size=3000)
            plt.title(f"Visualización del Nodo {node_id} y sus Vecinos")
            plt.show()
        else:
            print(f"El nodo con ID {node_id} no existe en el grafo.")

    def visualize_grid(self):
        posiciones = {}
        step_x = 1  # Espacio horizontal entre nodos
        step_y = -1  # Espacio vertical entre nodos
        start_pos = (0, 0)  # Posición inicial en el canvas

        for i, node in enumerate(self.G.nodes()):
            col = i // self.rows
            row = i % self.rows
            posiciones[node] = (start_pos[0] + col * step_x, start_pos[1] + row * step_y)

        nx.draw(self.G, posiciones, with_labels=True, node_color='lightgreen', node_size=500)

        plt.title("Grafo Personalizado en Forma de Grid")
        plt.show()
    
    def visualize_path(self, v_path):
        posiciones = {}
        step_x = 1  # Espacio horizontal entre nodos
        step_y = -1  # Espacio vertical entre nodos
        start_pos = (0, 0)  # Posición inicial en el canvas
        path = [v_path[n] for n in range(len(v_path)) if n % 2 == 0 ]

        for i, node in enumerate(self.G.nodes()):
            col = i // self.rows
            row = i % self.rows
            posiciones[node] = (start_pos[0] + col * step_x, start_pos[1] + row * step_y)

        # Dibujar el grafo completo
        nx.draw(self.G, posiciones, with_labels=True, node_color='gray', node_size=500, edge_color='gray')

        # Resaltar la ruta
        path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
        nx.draw_networkx_nodes(self.G, posiciones, nodelist=path, node_color='red', node_size=700)
        nx.draw_networkx_edges(self.G, posiciones, edgelist=path_edges, edge_color='red', width=2)

        plt.title("Ruta en el Grafo")
        plt.show()

class algorithms():
    def __init__(self):
        pass
    
    def calculate_node_value(self,node,v_nodes,grafo):
        values = np.array([1.0,1.0,1.0,1.0]) # inicializamos los valores de los 4 caminos posibles (N,S,E,W) de cada estado
        i = np.array([np.inf,np.inf,np.inf,np.inf]) # 
        next_node = [grafo.deadend,grafo.deadend,grafo.deadend,grafo.deadend]
        
        for neighbor in grafo.G[node]:
            for edge in grafo.G[node][neighbor]:
                if grafo.G[node][neighbor][edge]['action'] == 'N':
                    values[0] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*0.9
                    i[0] = 1
                    if grafo.deadend != neighbor:
                        next_node[0] = neighbor
                if grafo.G[node][neighbor][edge]['action'] == 'S':
                    values[1] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*0.9
                    i[1] = 1
                    if grafo.deadend != neighbor:
                        next_node[1] = neighbor
                if grafo.G[node][neighbor][edge]['action'] == 'E':
                    values[2] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*0.9
                    i[2] = 1
                    if grafo.deadend != neighbor:
                        next_node[2] = neighbor
                if grafo.G[node][neighbor][edge]['action'] == 'W':
                    values[3] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*0.9
                    i[3] = 1
                    if grafo.deadend != neighbor:
                        next_node[3] = neighbor
                if grafo.G._node[node]['goal']:
                    values[0] = 0

        return min(values*i), np.argmin(values*i), next_node[np.argmin(values*i)] # (menor valor, direccion, siguiente nodo)

    def inicialization(self,grafo):
        # inicializamos los valores de los estados del grafo
        v_nodes = {} # creamos un diccionario que almacene los valores de los estados del grafo
        for node_id,node in grafo.G.nodes.items():
            v_nodes[node_id] = abs(int(node_id) - int(grafo.goal))*1.5
        return v_nodes

    def decode_path(self,meta,path,nodo_id,dic_path= {0:'↑', 1:'↓', 2:'→', 3:'←'}):
        if nodo_id == int(meta) -1:
            lista = []
            lista.append(meta)
            return lista
        else:
            _nodo_id = int(path[nodo_id][1]) - 1
            tup = str(nodo_id + 1),dic_path[path[nodo_id][0]]
            lista = list(tup) + self.decode_path(meta,path,_nodo_id,dic_path)
            
            return lista
        
    def iteration_value_alg(self,grafo,delta=0.01):
        v_nodes = self.inicialization(grafo)
        print(f'Nodos inicializados: \n{v_nodes}\n')

        table = []
        table_arrows = []
        arrows_dict = {0:'↑', 1:'↓', 2:'→', 3:'←'}
        table.append(list(v_nodes.values()))
        _delta = np.inf
        iteracion = 0
        
        while True:
            path = []
            arrows = []
            current_values = list(v_nodes.values())
            for key, value in v_nodes.items():
                v_nodes[key], direction, next_node = self.calculate_node_value(str(key),v_nodes,grafo)
                path.append((direction, next_node))
                arrows.append(arrows_dict[direction])
            _delta = np.abs(np.array(current_values)-np.array(list(v_nodes.values())))
            table.append(list(v_nodes.values()))
            table_arrows.append(arrows)
            iteracion += 1
            # print(list(v_nodes.values()))

            if max(_delta[:-1]) < delta:
                print(f'\nNúmero de iteraciones: {iteracion}\n')
                break
        
        nodo_inicial = int(grafo.start) - 1
        decoded_path = self.decode_path(grafo.goal,path,nodo_inicial)

        return np.array(table), np.array(table_arrows), decoded_path

