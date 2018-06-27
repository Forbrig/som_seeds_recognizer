import random
from util import *
import numpy as np

# neuron class
class neuron:
    def __init__(self, tamanho, aleatorio):
        self.tamanho = tamanho
        if aleatorio == True:
            self.pesos = self.pesos_aleatorios(tamanho)
        else:
            self.pesos = np.array(tamanho[0] * [tamanho[1] * [1]])

    # generate random wights
    def pesos_aleatorios(self, tamanho):
        pesos = []
        for i in range(tamanho[0]):
            linha = []
            for j in range(tamanho[1]):
                linha.append(random.uniform(0, 1))
            pesos.append(linha)
        return np.array(pesos)

    def pesos_aleatorios2(self, tamanho):
        lista = [random.uniform(0, 1),
                 random.uniform(0, 1),
                 random.uniform(0, 1),
                 random.uniform(0, 1),
                 random.uniform(0, 1),
                 random.uniform(0, 1)]
        return np.array([lista])

    # calculate the distance between 2 neurons (euclidian distance)
    def soma_de_pesos(self, entrada):
        return sum(sum((entrada - self.pesos) ** 2))
