from navigator import Own_graph, algorithms
import numpy as np

''' 
- Instanciamos la clase own_graph con los datos del json
- Instanciamos la clase agoritmos donde se han implementado los algortitmos de 
  Iteración de Valor e Iteración de Política
'''
grafo = Own_graph('./test_cases/navigator3-15-0-0.json')
algo = algorithms()

# grafo.visualize_grid()

# Ejecutamos el algoritmo de iteración de valor
algo.iteration_value_alg(grafo,0.01)

np.set_printoptions(threshold=np.inf, linewidth=300, edgeitems=10)

# Guardamos los resultados de las iteraciones en el archivo 'Resultados.log'
with open('Resultados.log', 'w+', encoding='utf-8') as f:
  # Escribimos en el log
  f.write(f'#### Algoritmo de Iteración de Valor ###\n------------------------------------------\n')
  f.write(f'\nNodos inicializados:\n\n{algo.iv_result['node_init_values']}\n')
  f.write(f'\nNúmero de iteraciones: {algo.iv_result['num_iterations']}\n')
  f.write(f'\nTiempo de ejecución: {algo.iv_result['total_time']} s\n')
  f.write(f'\nPolítica final: \n\n{np.reshape(algo.iv_result['table_arrows'][-1][:-1],(grafo.columns,grafo.rows))}\n')

  f.write(f'\nDetalle de las iteraciones\n-------------------------\n')
  f.write('\nTabla de valores por cada iteración:\n\n')
  for fila in algo.iv_result['table_values']:
      f.write(f'{fila}\n')

  f.write('\nTabla de direcciones por cada iteración:\n\n')
  for fila in algo.iv_result['table_arrows']:
      f.write(f'{fila}\n')

# grafo.visualize_path(path)
