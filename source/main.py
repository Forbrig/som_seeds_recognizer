import matplotlib.pyplot as plt
import numpy as np
import time

from grid import *
from util import *

########################## CONFIG ##########################

carregar_treino = "output/toda-entrada.tra.out"         # Arquivo de treino armazenado anteriormente
treinar         = "input/seeds_dataset.txt"           # Arquivo de entrada para treinar
teste_de_acerto = "input/seeds_dataset.txt"         # Arquivo de entrada para testar taxa de acerto
teste_de_treino = "input/seeds_dataset.txt"           # Arquivo de entrada gerar a grade de reconhecimento
n_iter          = 20                                     # Quantidade de iterações de treino

TREINAR                         = True
GERAR_GRADE_RECONHECIMENTO      = True
CALCULAR_TAXA_ACERTOS           = True

GRID_SIZE_I                     = 6
GRID_SIZE_J                     = 6

ALPHA_INICIAL                   = 0.1
DECRE_ALPHA                     = 0.00009

SIGMA_INICIAL                   = 2
DECRE_SIGMA                     = 0.0015
###############################################################################

t0 = time.time()
print("Reading input")
aux_entrada, aux_rotulo = read_file(teste_de_acerto)
entrada_acertos = []
ind = 0
for i in aux_entrada:
    entrada_acertos.append([i, aux_rotulo[ind]])
    ind += 1


entrada_treino = read_file(treinar)
aux_entrada, aux_rotulo = read_file(teste_de_treino)
entrada_teste = []
ind = 0
for i in aux_entrada:
    entrada_teste.append([i, aux_rotulo[ind]])
    ind += 1


print("Generating neuron grid")
g = Grade((GRID_SIZE_I, GRID_SIZE_J), (1, 7), ALPHA_INICIAL, DECRE_ALPHA, SIGMA_INICIAL, DECRE_SIGMA)
#def __init__(self, tam_grade, tam_entrada, alpha, taxa, sigma, s_taxa):


if TREINAR:
    print("Training")
    ti = time.time()
    for i in range(n_iter):
        print("Epoch: ", i, "| alpha: %.6f"% g.alpha, "| sigma: %.6f"% g.sigma)
        g.treinar(entrada_treino[0])
        g.alpha -= g.taxa
        g.sigma -= g.s_taxa
    tf = time.time()
    print()
    print("Training time: %fs"%(tf - ti))

if GERAR_GRADE_RECONHECIMENTO:
    print("Generating recognizer grid")
    g.grade_de_reconhecimento(entrada_teste)
    plot_grid(g.rec_grid, GRID_SIZE_I, GRID_SIZE_J)


if CALCULAR_TAXA_ACERTOS:
    print("Calculating hit rate")
    taxa_de_acerto = g.reconhece_lista(entrada_acertos)
    print("Hit rate: %.4f" %(taxa_de_acerto))

print("Total time: %fs" %(time.time() - t0))
