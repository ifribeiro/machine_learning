import numpy

def calc_sse(centroids: numpy.ndarray, labels: numpy.ndarray, data: numpy.ndarray):
    """
    Calcula as somas dos quadrados dos erros
    """
    distances = 0
    for i, c in enumerate(centroids):
        idx = numpy.where(labels == i)
        dist = numpy.sum((data[idx] - c)**2)
        distances += dist
    return distances


class KMeans:
    """
    Algoritmo K-Means
    
    Params:
        - n_cluster : Número de clusters
        - init_pp : Método de inicialização (True usa KMeans++)
        - max_iter : Número máximo de interações para atualização do centroides
        - tolerance : Parâmatro de parada do algoritmo
        - seed : Seed aleatório
        - centroid : centroids 
        - SSE : Soma dos quadrados dos erros
    """

    def __init__(
            self,
            n_cluster: int,
            init_pp: bool = True,
            max_iter: int = 300,
            tolerance: float = 1e-4,
            seed: int = None):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.init_pp = init_pp
        self.seed = seed
        self.centroids = None
        self.SSE = None

    def fit(self, data: numpy.ndarray):
        """
        Obtem os centroids dos dados

        Params:
            - data: base de dados
        """
        self.centroids = self.inicializa_centroids(data)
        for _ in range(self.max_iter):
            distance = self.calc_distance(data)
            cluster = self.atribuir_cluster(distance)
            new_centroid = self.atualiza_centroids(data, cluster)
            diff = numpy.abs(self.centroids - new_centroid).mean()
            self.centroids = new_centroid

            if diff <= self.tolerance:
                break

        self.SSE = calc_sse(self.centroids, cluster, data)

    def predict(self, data: numpy.ndarray):
        """
        Realiza a predição dos clusters de uma base de dados

        Params:
        - data : base de dados
        """
        distance = self.calc_distance(data)
        cluster = self.atribuir_cluster(distance)
        return cluster

    def inicializa_centroids(self, data: numpy.ndarray):
        """
        Inicializa centroids usando o método KMeans++

        Params:
        - data: amostra dos dados

        """
        if self.init_pp:
            numpy.random.seed(self.seed)
            centroids = [int(numpy.random.uniform()*len(data))]
            for _ in range(1, self.n_cluster):
                dist = []
                dist = [min([numpy.inner(data[c]-x, data[c]-x) for c in centroids])
                        for i, x in enumerate(data)]
                dist = numpy.array(dist)
                dist = dist / dist.sum()
                cumdist = numpy.cumsum(dist)

                prob = numpy.random.rand()
                for i, c in enumerate(cumdist):
                    if prob > c and i not in centroids:
                        centroids.append(i)
                        break
            centroids = numpy.array([data[c] for c in centroids])
        else:
            numpy.random.seed(self.seed)
            idx = numpy.random.choice(range(len(data)), size=(self.n_cluster))
            centroids = data[idx]
        return centroids

    def calc_distance(self, data: numpy.ndarray):
        """
        Calcula distância entre os dados e centroids
        Params:
            - data : dados a serem utilizados nos calculos da distância
        """
        distances = []
        for c in self.centroids:
            distance = numpy.sum((data - c) * (data - c), axis=1)
            distances.append(distance)

        distances = numpy.array(distances)
        distances = distances.T
        return distances

    def atribuir_cluster(self, distance: numpy.ndarray):
        """
        Atribui um cluster para os dados
        Params:
         - distance : distância de cada dados para os centroids
        """
        cluster = numpy.argmin(distance, axis=1)
        return cluster

    def atualiza_centroids(self, data: numpy.ndarray, cluster: numpy.ndarray):
        """
        Atualiza centroids com base na media de cada cluster dos dados
        Params:
         - data : dados para obtenção das médias
         - cluster : cluster para cada dado
        """
        centroids = []
        for i in range(self.n_cluster):
            idx = numpy.where(cluster == i)
            centroid = numpy.mean(data[idx], axis=0)
            centroids.append(centroid)
        centroids = numpy.array(centroids)
        return centroids


if __name__ == "__main__":
    pass