from neuron import *
from util import most_common
from math import e
import time

# neuron grid:
# size of neuron grid, size of input, learn rate, learn rate decrease, neighbor sigma, neighbor sigma decrease
class grid:
    def __init__(self, tam_grade, tam_entrada, alpha, taxa, sigma, s_taxa):
        self.taxa = taxa
        self.sigma = sigma
        self.s_taxa = s_taxa
        self.alpha = alpha
        self.tam_grade = tam_grade
        self.tam_entrada = tam_entrada
        self.grade = self.grade_aleatoria(tam_grade, tam_entrada)
        self.BMU = None
        self.neighbors_size = 8
        self.rec_grid = None

    # recognize (classify) input
    def classify(self, entrada):
        #   Encontra o BMU
        indice = self.reconhece2(entrada)[0]

        #   Tenta pegar o valor do número reconhecido
        try:
            d = self.rec_grid[indice]
        #   Se não conseguiu, é porque é um erro
        except:
            d = -1
        return d

    # Calculating hit rate of a input list
    def reconhece_lista(self, lista_de_teste):
        hits = 0
        for i in lista_de_teste:
            d = self.classify(i[0])
            if d == i[1]:
                hits += 1
        #   Retorna a porcentagem de acertos
        return ((hits / len(lista_de_teste)) * 100)

    # generate the recognizer grid, the input is a list of examples
    # that tells the neuron which class he most recognize or is more like
    def grade_de_reconhecimento(self, lista_de_teste):
        desenhar = {}
        # Reconhece a entrada
        for i in lista_de_teste:
            z = self.reconhece2(i[0])
            try:
                desenhar[z[0]].append(i[1])
            except:
                desenhar[z[0]] = [i[1]]
        # Reconhece o mais frequente
        for i in desenhar:
            desenhar[i] = most_common(desenhar[i])

        self.rec_grid = desenhar


    # return the BMU for that input
    def reconhece2(self, entrada):
        return self.melhor_neuronio(entrada)

    # do the training for a input list
    def treinar(self, entrada_de_treino):
        alpha = self.alpha
        c = 0
        a = len(entrada_de_treino)
        for i in entrada_de_treino:
            print("%d/%d" %(c, a), end = "\r")
            self.iteracao(i, alpha)
            c += 1
        print("%d/%d" %(c, a))

    def repesa_neuronio(self, indice_neuronio, entrada, alpha):
        neuronio = self.grade[indice_neuronio[0]][indice_neuronio[1]]
        return neuronio.pesos + self.vizinhanca(indice_neuronio) * alpha * (entrada - neuronio.pesos)

    # do the training for a single input
    # update the BMU and all his neighbors
    def iteracao(self, entrada, alpha):
        #   Encontra o BMU
        self.BMU = self.melhor_neuronio(entrada)
        #   Repesa a vizinhança
        for i in self.vizinhos(self.BMU[0]):
            self.grade[i[0]][i[1]].pesos = self.repesa_neuronio((i[0], i[1]), entrada, alpha)

    # return a list of neighbors of a neuron
    def vizinhos2(self, x):
        l = [x]
        for i in range(self.tam_grade[0]):
            for j in range(self.tam_grade[1]):
                l.append((i, j))
        self.neighbors_size = len(l)
        return l

    # neighbor function
    # calculate using sigma
    def vizinhanca(self, indice_neuronio):
        neuronio = self.grade[indice_neuronio[0]][indice_neuronio[1]]
        melhor = self.grade[self.BMU[0][0]][self.BMU[0][1]]
        S = neuronio.soma_de_pesos(melhor.pesos)
        return e ** ((-(S ** 2)) / (2 * ((self.tam_entrada[0] * self.tam_entrada[1]) ** 2)))

    # generate the inital grid of neurons with random weights
    def grade_aleatoria(self, tam_grade, tam_entrada):
        grade = []
        for i in range(tam_grade[0]):
            linha = []
            for j in range(tam_grade[1]):
                linha.append(neuron(tam_entrada, True))
            grade.append(linha)
        return grade

    # find BMU: smallest euclidian distance betwen the input and all the neurons
    def melhor_neuronio(self, entrada):
        tam_grade = self.tam_grade
        melhor = ((0, 0), self.grade[0][0].soma_de_pesos(entrada))
        for i in range(0, tam_grade[0]):
            for j in range(0, tam_grade[1]):
                sd = self.grade[i][j].soma_de_pesos(entrada)
                if sd < melhor[1]:
                    melhor = ((i, j), sd)
        return melhor

    # return the neighbors of a neuron in a list
    def vizinhos(self, pos):
        vizinhos = []
        tam_grade = self.tam_grade

        if pos[0] + 1 in range(tam_grade[0]) and pos[1] in range(tam_grade[1]):
            vizinhos.append([pos[0] + 1, pos[1]])

        if pos[0] in range(tam_grade[0]) and pos[1] + 1 in range(tam_grade[1]):
            vizinhos.append([pos[0], pos[1] + 1])

        if pos[0] in range(tam_grade[0]) and pos[1] - 1 in range(tam_grade[1]):
            vizinhos.append([pos[0], pos[1] - 1])

        if pos[0] - 1 in range(tam_grade[0]) and pos[1] + 1 in range(tam_grade[1]):
            vizinhos.append([pos[0] - 1, pos[1] + 1])

        if pos[0] - 1 in range(tam_grade[0]) and pos[1] - 1 in range(tam_grade[1]):
            vizinhos.append([pos[0] - 1, pos[1] - 1])

        if pos[0] + 1 in range(tam_grade[0]) and pos[1] - 1 in range(tam_grade[1]):
            vizinhos.append([pos[0] + 1, pos[1] - 1])

        if pos[0] + 1 in range(tam_grade[0]) and pos[1] - 1 in range(tam_grade[1]):
            vizinhos.append([pos[0] + 1, pos[1] - 1])

        if pos[0] - 1 in range(tam_grade[0]) and pos[1] in range(tam_grade[1]):
            vizinhos.append([pos[0] - 1, pos[1]])

        self.neighbors_size = len(vizinhos)
        return vizinhos
