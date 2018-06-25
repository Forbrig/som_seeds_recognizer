import matplotlib.pyplot as plt
import numpy as np
import time

from grid import *
from util import *

########################## VARIAVEIS DE CONFIGURAÇÂO ##########################

carregar_treino = "output/toda-entrada.tra.out"         # Arquivo de treino armazenado anteriormente
treinar         = "input/seeds_dataset.txt"           # Arquivo de entrada para treinar
teste_de_acerto = "input/seeds_dataset.txt"         # Arquivo de entrada para testar taxa de acerto
teste_de_treino = "input/seeds_dataset.txt"           # Arquivo de entrada gerar a grade de reconhecimento
n_iter          = 20                                     # Quantidade de iterações de treino

CARREGAR_TREINO                 = False
TREINAR                         = True
GERAR_GRADE_RECONHECIMENTO      = True
CALCULAR_TAXA_ACERTOS           = True

GRID_SIZE_I                     = 8
GRID_SIZE_J                     = 4

ALPHA_INICIAL                   = 0.1
DECRE_ALPHA                     = 0.00009

SIGMA_INICIAL                   = 2
DECRE_SIGMA                     = 0.0015
###############################################################################

t0 = time.time()
print("> Lendo arquivo de entrada")
aux_entrada, aux_rotulo = leitura_entrada(teste_de_acerto)
entrada_acertos = []
ind = 0
for i in aux_entrada:
    entrada_acertos.append([i, aux_rotulo[ind]])
    ind += 1


entrada_treino = leitura_entrada(treinar)
aux_entrada, aux_rotulo = leitura_entrada(teste_de_treino)
entrada_teste = []
ind = 0
for i in aux_entrada:
    entrada_teste.append([i, aux_rotulo[ind]])
    ind += 1


print("> Gerando grade de neurônios")
g = Grade((GRID_SIZE_I, GRID_SIZE_J), (1, 7), ALPHA_INICIAL, DECRE_ALPHA, SIGMA_INICIAL, DECRE_SIGMA)
#def __init__(self, tam_grade, tam_entrada, alpha, taxa, sigma, s_taxa):


if CARREGAR_TREINO:
    print("> Carregando treino")
    carregar(g, carregar_treino)

if TREINAR:
    print("> Treinando")
    ti = time.time()
    for i in range(n_iter):
        print("> Época: ", i, "| alpha: %.6f"% g.alpha, "| sigma: %.6f"% g.sigma)
        g.treinar(entrada_treino[0])
        g.alpha -= g.taxa
        g.sigma -= g.s_taxa
    tf = time.time()
    print()
    print(">>> Tempo de treinamento: %fs"%(tf - ti))

if GERAR_GRADE_RECONHECIMENTO:
    print("> Gerando grade de reconhecimento")
    g.grade_de_reconhecimento(entrada_teste)
    print(g.rec_grid)
    for x in range(0, GRID_SIZE_I):
        for y in range(0, GRID_SIZE_J):
            if g.rec_grid[(x,y)] == 3:
                plt.scatter(x, y, c='blue')
            elif g.rec_grid[(x,y)] == 2:
                plt.scatter(x, y, c='red')
            elif g.rec_grid[(x,y)] == 1:
                plt.scatter(x, y, c='green')
    plt.show()

if CALCULAR_TAXA_ACERTOS:
    print("> Calculando taxa de acertos")
    taxa_de_acerto = g.reconhece_lista(entrada_acertos)
    print("  - Taxa de acerto: %.4f" %(taxa_de_acerto))

print("# Tempo total: %fs" %(time.time() - t0))
