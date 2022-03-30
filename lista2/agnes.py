
import numpy as np
import pandas as pd
import re

def dist_euclidiana(x, y):
    return np.linalg.norm(x-y)

def distancia_minima(first_d, src_houses, dest_houses):
    """
    Obtem a distância minina
    """
    min_value = float("inf")
    for s_house in src_houses:
        for d_house in dest_houses:
            if first_d[int(s_house) - 1][int(d_house) - 1] < min_value:
                min_value = first_d[int(s_house) - 1][int(d_house) - 1]
    return min_value

def atualiza_distancia(distance_matrix, first_d, tag):
    """
    Atualiza distâncias da matriz
    """
    src_houses = re.split(',', tag)
    for column in distance_matrix.columns:
        if column == tag:
            distance_matrix[column][column] = float("inf")
        else:
            dest_houses = re.split(',', column)
            distance_matrix[tag][column] = distance_matrix[column][tag] = distancia_minima(first_d, src_houses, dest_houses)
    return distance_matrix

def calc_distancia_matriz(m, data):
    """
    Calcula matriz de distância dos dados
    """
    d = np.zeros((m, m))
    for i in range(len(d)):
        for j in range(len(d)):
            if i == j:
                d[i][j] = 0
            else:
                d[i][j] = dist_euclidiana(data[i, :], data[j, :])
    # cria dataframe pandas
    df_matrix = pd.DataFrame(d, [i.__str__() for i in range(1, m + 1)],
                                   [i.__str__() for i in range(1, m + 1)])

    # atribui infinito para valores onde i==j
    # facilita o calculo da distancia minima
    d[d == 0.0] = float('inf')
    return df_matrix, d


def agnes(dataset, n_clusters=2, v=False):
    """
    Agrupa os dados utilizando o algoritmo hierarquico AGNES
    """
    len_data = len(dataset)
    if v:
        print ("Calculando matriz de distâncias.")
    
    # Calcula matriz de distâncias
    matriz_distancias, d = calc_distancia_matriz(len_data, dataset)
    if v:
        print ("Obtendo clusters.")

    while len(matriz_distancias.columns) > n_clusters:
        # menor distancia na matriz
        min_d = min(matriz_distancias.min())

        # indice da distancia minima no dataframe
        temp_index = [
            (matriz_distancias[col][matriz_distancias[col] == min_d].index[i], matriz_distancias.columns.get_loc(col))
            for col
            in matriz_distancias.columns for i in
            range(len(matriz_distancias[col][matriz_distancias[col] == min_d].index))]

        index = [temp_index[0][0], matriz_distancias.columns[temp_index[0][1]]]

        # atualiza os indices
        indexes = np.array(matriz_distancias.index)
        new_indexes = [i.__str__() for i in indexes]
        new_indexes[new_indexes.index(index[0].__str__())] = index[0].__str__() + ',' + index[1].__str__()
        matriz_distancias = matriz_distancias.reindex(new_indexes)

        # atualiza colunas com base nos indices
        matriz_distancias.columns = new_indexes
        # Remove colunas e linhas extras
        matriz_distancias.drop(columns=index[1], inplace=True)
        matriz_distancias.drop(index=index[1], inplace=True)

        # atualiza a matriz de distâncias com a nova matriz
        matriz_distancias = atualiza_distancia(matriz_distancias, d, index[0].__str__() + ',' + index[1].__str__())
    cluster_labels = np.zeros(len_data)

    for j in range(len(matriz_distancias.columns)):
        for i in re.split(',', matriz_distancias.columns[j]):
            cluster_labels[int(i) - 1] = j + 1

    return cluster_labels


if __name__ == "__main__":
    pass