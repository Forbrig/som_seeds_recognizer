import matplotlib.pyplot as plt
import numpy as np
import time

from grid import *
from util import *

########################## CONFIG ##########################

treinar         = "input/train.in" # dataset for training
teste_de_treino = "input/train.in"# data set to generate the recognizer grid
teste_de_acerto = "input/test.in" # dataset to test the hit rate

n_iter          = 20 # number of iterations

train           = True
recognizer_grid = True
hit_rate        = True

GRID_SIZE_I     = 4
GRID_SIZE_J     = 4

ALPHA_INICIAL   = 0.2
DECRE_ALPHA     = 0.00009

SIGMA_INICIAL   = 2
DECRE_SIGMA     = 0.0015

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

print("Generating neuron grid...")
g = grid((GRID_SIZE_I, GRID_SIZE_J), (1, 7), ALPHA_INICIAL, DECRE_ALPHA, SIGMA_INICIAL, DECRE_SIGMA)

if train:
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

if recognizer_grid:
    print("Generating recognizer grid")
    g.grade_de_reconhecimento(entrada_teste)
    plot_grid(g.rec_grid, GRID_SIZE_I, GRID_SIZE_J)


if hit_rate:
    print("Calculating hit rate")
    taxa_de_acerto = g.reconhece_lista(entrada_acertos)
    print("Hit rate: %.4f" %(taxa_de_acerto))

print("Total time: %fs" %(time.time() - t0))
