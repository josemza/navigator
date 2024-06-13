import json
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import timeit
from functools import partial
from memory_profiler import profile, memory_usage
import copy
import time


class Node:
    def __init__(self, id, goal, deadend, heuristic):
        """
        Inicializa un nodo con su ID, si es el objetivo, el deadend y su heurística.
        """
        self.id = id
        self.goal = goal
        self.deadend = deadend
        self.heuristic = heuristic
        self.adjacencies = []  # List of adjacencies

    def add_adjacency(self, node_name, actions):
        """
        Añade una adyacencia al nodo.
        
        Parameters:
        - node_name: Nombre del nodo adyacente.
        - actions: Acciones posibles y sus probabilidades.
        """
        self.adjacencies.append((node_name, actions))

    def __repr__(self):
        """
        Representación en cadena del nodo.
        """
        return f"Node({self.id}, Goal={self.goal}, DeadEnd={self.deadend}, Heuristic={self.heuristic}, Adjacencies={self.adjacencies})"

class Own_graph:
    def __init__(self, json_file):
        """
        Inicializa el grafo a partir de un archivo JSON.
        
        Parameters:
        - json_file: Ruta al archivo JSON que define el grafo.
        """
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
        """
        Carga los nodos desde el archivo JSON y los inicializa.
        """
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
        """
        Crea el grafo a partir de los nodos cargados.
        """
        # Añadir nodos
        for node_id, node in self.nodes.items():
            self.G.add_node(node_id, goal=node.goal, deadend=node.deadend, heuristic=node.heuristic)

        # Añadir aristas con pesos y probabilidades
        for node_id, node in self.nodes.items():
            for adj_name, actions in node.adjacencies:
                for action, prob in actions.items():
                    self.G.add_edge(node_id, adj_name, action=action, weight=1, probability=prob)

    def visualize_node_and_neighbors(self, node_id):
        """
        Visualiza un nodo y sus vecinos en el grafo.
        
        Parameters:
        - node_id: ID del nodo a visualizar.
        """
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
        """
        Visualiza el grafo en forma de una cuadrícula.
        """
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
        """
        Visualiza una ruta en el grafo.
        
        Parameters:
        - v_path: Lista de nodos que representan la ruta a visualizar.
        """
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

class algorithms:
    def __init__(self):
        """
        Inicializa la clase de algoritmos, limpiando los logs y preparando las estructuras de resultados.
        """
        self.clean_log('Memory_usage.log','Resultados.log','Resultados2.log')
        self.iv_result = {}
        self.ip_result = {}
    
    def calculate_node_value(self,node,v_nodes,grafo):
        """
        Calcula el valor de un nodo basado en los valores de sus vecinos y las probabilidades de transición.
        
        Parameters:
        - node: Nodo actual.
        - v_nodes: Diccionario con los valores de los nodos.
        - grafo: Instancia del grafo.

        Returns:
        - Menor valor calculado.
        - Dirección óptima.
        - Siguiente nodo óptimo.
        """
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

    def initialization(self,grafo):
        """
        Inicializa los valores de los estados del grafo.

        Parameters:
        - grafo: Instancia del grafo.

        Returns:
        - v_nodes: Diccionario que almacena los valores iniciales de los estados del grafo.
        """
        v_nodes = {} # Diccionario para almacenar los valores de los estados del grafo
        for node_id,node in grafo.G.nodes.items():
            v_nodes[node_id] = abs(int(node_id) - int(grafo.goal))*1.5
        
        return v_nodes

    def decode_path(self,meta,path,nodo_id,dic_path= {0:'↑', 1:'↓', 2:'→', 3:'←'}):
        """
        Decodifica el camino desde el nodo inicial hasta la meta.

        Parameters:
        - meta: Nodo meta.
        - path: Camino representado como una lista de tuplas (dirección, siguiente nodo).
        - nodo_id: ID del nodo actual.
        - dic_path: Diccionario para traducir las direcciones (default: {0: '↑', 1: '↓', 2: '→', 3: '←'}).

        Returns:
        - lista: Lista de nodos y direcciones que representan el camino decodificado.
        """
        if nodo_id == int(meta) -1:
            lista = []
            lista.append(meta)
            return lista
        else:
            _nodo_id = int(path[nodo_id][1]) - 1
            tup = str(nodo_id + 1),dic_path[path[nodo_id][0]]
            lista = list(tup) + self.decode_path(meta,path,_nodo_id,dic_path)
            
            return lista
    
    @profile(stream=open('Memory_usage.log','w+', encoding='utf-8'))
    def iteration_value_alg(self,grafo,delta=0.01):
        """
        Ejecuta el algoritmo de iteración de valor para determinar los valores y políticas óptimas de los nodos.

        Parameters:
        - grafo: Instancia del grafo.
        - delta: Umbral de convergencia (default: 0.01).
        """
        start_time = time.time()

        # Inicializamos los nodos con el metodo
        v_nodes = self.initialization(grafo)

        # Almacenamos los nodos iniciales en el diccionario de resultados
        self.iv_result['node_init_values'] = copy.deepcopy(v_nodes)

        # Definimos variables
        table = [] # Almacena todos los valores que se van calculando para cada estado
        policy = [] # Almacena todos las direcciones que se van calculando para cada estado
        arrows_dict = {0:'↑', 1:'↓', 2:'→', 3:'←'} # Diccionario para traducir las direccones
        _delta = np.inf
        iteracion = 0 # Contador para las iteraciones
        
        # Agregamos a la tabla de resultados los valores de inicializacion de los estados
        table.append(list(v_nodes.values()))

        # Algoritmo iteracion de valor
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
            policy.append(arrows)
            iteracion += 1

            if max(_delta[:-1]) < delta:
                break
        
        end_time = time.time()

        nodo_inicial = int(grafo.start) - 1

        self.iv_result['path'] = self.decode_path(grafo.goal,path,nodo_inicial)
        self.iv_result['table_values'] = np.array(table)
        self.iv_result['policy'] = np.array(policy)
        self.iv_result['num_iterations'] = iteracion
        self.iv_result['total_time'] = end_time - start_time

        # return self.iv_result

    def write_log(self,message):
        """
        Escribe un mensaje en el archivo de log de resultados.

        Parameters:
        - message: Mensaje a escribir en el log.
        """
        with open('Resultados.log', 'w+', encoding='utf-8') as f:
            f.write(message)
    
    def clean_log(self,*args):
        """
        Limpia los archivos de log especificados.

        Parameters:
        - *args: Nombres de los archivos de log a limpiar.
        """
        for arg in args:
            open(arg, 'w', encoding='utf-8')
    
    # Inicializacion de politicas
    def initialization_policy(self, grafo):
        """
        Inicializa las políticas del grafo.

        Parameters:
        - grafo: Instancia del grafo.

        Returns:
        - pol: Diccionario con la política inicial para cada nodo.
        """
        pol = {}
        for nodo in grafo.G.nodes():
            for neighborg in grafo.G[nodo]:
                if neighborg != grafo.deadend:
                    pol[nodo] = grafo.G[nodo][neighborg][0]['action']
                    break
        pol[grafo.deadend] = grafo.G[grafo.deadend][neighborg][0]['action']

        return pol

    # Evaluación de las políticas
    def evaluate_policy(self, grafo, v_nodes, policy, discount_factor=0.9, theta=0.01):
        """
        Evalúa la política actual del grafo.

        Parameters:
        - grafo: Instancia del grafo.
        - v_nodes: Diccionario con los valores de los nodos.
        - policy: Política actual.
        - discount_factor: Factor de descuento (default: 0.9).
        - theta: Umbral de convergencia (default: 0.01).

        Returns:
        - v_nodes: Diccionario actualizado con los valores de los nodos.
        """
        value = 1
        # while True:
        delta = 0
        for node in v_nodes.keys():
            val_ini = 1
            v = v_nodes[node]
            action = policy[node]
            # v_nodes[node] = sum(
            #     grafo.G[node][neighbor][action]['probability'] * (1 + discount_factor * v_nodes[neighbor])
            #     for neighbor in grafo.G[node]
            #     if action in grafo.G[node][neighbor]
            # )
            for neighbor in grafo.G[node]:
                for edge in grafo.G[node][neighbor]:
                    if grafo.G[node][neighbor][edge]['action'] == action:
                        val_ini += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*discount_factor
                        # if grafo.deadend != neighbor:
                        #     next_node[0] = neighbor
            
            v_nodes[node] = val_ini

            if grafo.G._node[node]['goal']:
                v_nodes[node] = 0

            delta = max(delta, abs(v - v_nodes[node]))
            if delta < theta:
                break
        return v_nodes

    def improve_policy(self, grafo, v_nodes, policy, discount_factor=0.9):
        """
        Mejora la política actual del grafo.

        Parameters:
        - grafo: Instancia del grafo.
        - v_nodes: Diccionario con los valores de los nodos.
        - policy: Política actual.
        - discount_factor: Factor de descuento (default: 0.9).

        Returns:
        - tuple: (policy, v_nodes, policy_stable)
          - policy: Política mejorada.
          - v_nodes: Diccionario actualizado con los valores de los nodos.
          - policy_stable: Booleano indicando si la política es estable.
        """
        policy_stable = True
        # for node in v_nodes.keys():
        #     old_action = policy[node]
        #     action_values = {}
        #     for neighbor in grafo.G[node]:
        #         for action in grafo.G[node][neighbor]:
        #             q_value = grafo.G[node][neighbor][action]['probability'] * (1 + discount_factor * v_nodes[neighbor])
        #             action_values[action] = action_values.get(action, 0) + q_value
        #     best_action = max(action_values, key=action_values.get)
        #     policy[node] = best_action
            
        #     if old_action != best_action:
        #         policy_stable = False

        # next_node = [grafo.deadend,grafo.deadend,grafo.deadend,grafo.deadend]
        actions_dict = {'N':0, 'S':1, 'E':2, 'W':3} # Diccionario para traducir las direccones

        old_action = copy.deepcopy(policy)

        for node in v_nodes.keys():
            values = np.array([1.0,1.0,1.0,1.0]) # inicializamos los valores de los 4 caminos posibles (N,S,E,W) de cada estado
            i = np.array([np.inf,np.inf,np.inf,np.inf]) # 
            for neighbor in grafo.G[node]:
                for edge in grafo.G[node][neighbor]:
                    if grafo.G[node][neighbor][edge]['action'] == 'N':
                        values[0] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*discount_factor
                        i[0] = 1
                        # if grafo.deadend != neighbor:
                        #     next_node[0] = neighbor
                    if grafo.G[node][neighbor][edge]['action'] == 'S':
                        values[1] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*discount_factor
                        i[1] = 1
                        # if grafo.deadend != neighbor:
                        #     next_node[1] = neighbor
                    if grafo.G[node][neighbor][edge]['action'] == 'E':
                        values[2] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*discount_factor
                        i[2] = 1
                        # if grafo.deadend != neighbor:
                        #     next_node[2] = neighbor
                    if grafo.G[node][neighbor][edge]['action'] == 'W':
                        values[3] += grafo.G[node][neighbor][edge]['probability']*v_nodes[neighbor]*discount_factor
                        i[3] = 1
                        # if grafo.deadend != neighbor:
                        #     next_node[3] = neighbor
                    if grafo.G._node[node]['goal']:
                        values[0] = 0
                    
                    values[actions_dict[policy[node]]] = v_nodes[node]

            best_action = min(values*i)
            v_nodes[node] = best_action

            dir = [k for k, v in actions_dict.items() if v == list(values).index(best_action)]
            policy[node] = dir[0]

        if old_action != policy:
            policy_stable = False

        return policy, v_nodes, policy_stable

    @profile(stream=open('Memory_usage_2.log','w+', encoding='utf-8'))
    def iteration_policy_alg(self, grafo, discount_factor=0.9, theta=0.01):
        """
        Ejecuta el algoritmo de iteración de políticas para determinar las políticas y valores óptimos de los nodos.

        Parameters:
        - grafo: Instancia del grafo.
        - discount_factor: Factor de descuento (default: 0.9).
        - theta: Umbral de convergencia (default: 0.01).
        """
        start_time = time.time()
        
        arrows_dict = {'N':'↑', 'S':'↓', 'E':'→', 'W':'←'} # Diccionario para traducir las direccones
        table_values = []
        table_policies = []

        # Inicializamos los valores ed los nodos en cero
        v_nodes = self.initialization(grafo) # {node: 0 for node in grafo.G.nodes()}

        table_values.append(list(v_nodes.values()))

        # Initialize a random policy
        policy = self.initialization_policy(grafo)

        policy_arrows = [arrows_dict[dir] for dir in list(policy.values())]
        table_policies.append(policy_arrows)

        # Almacenamos los nodos iniciales en el diccionario de resultados
        self.ip_result['node_init_values'] = copy.deepcopy(policy)
        
        # Policy Iteration
        iter_count = 0
        while True:
            # Policy Evaluation
            v_nodes = self.evaluate_policy(grafo,v_nodes, policy, discount_factor, theta)
            
            # Policy Improvement
            policy, v_nodes, policy_stable = self.improve_policy(grafo, v_nodes, policy, discount_factor)

            table_values.append(list(v_nodes.values()))

            policy_arrows = [arrows_dict[dir] for dir in list(policy.values())]
            table_policies.append(policy_arrows)
            
            iter_count += 1
            if policy_stable:
                break

        end_time = time.time()
        
        # Storing the results
        self.ip_result['policy'] = np.array(table_policies)
        self.ip_result['table_values'] = np.array(table_values)
        self.ip_result['num_iterations'] = iter_count
        self.ip_result['total_time'] = end_time - start_time
        
        # return self.ip_result
