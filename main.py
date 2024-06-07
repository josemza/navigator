from navigator import Own_graph, algorithms

''' 
- Instanciamos la clase own_graph con los datos del json
- Instanciamos la clase agoritmos donde se han implementado los algortitmos de 
  Iteración de Valor e Iteración de Política
'''
grafo = Own_graph('./test_cases/navigator3-2-0-0.json')
algo = algorithms()

# grafo.visualize_grid()

# Ejecutamos el algoritmo de iteración de valor
table, table_arrows, path = algo.iteration_value_alg(grafo,0.01)

# Mostramos los resultados
print('Tabla de valores\n')
for fila in table:
    print(fila)
# print(table)

print('\nTabla de direcciones\n')
for fila in table_arrows:
    print(fila)
# print(table_arrows)

# grafo.visualize_path(path)