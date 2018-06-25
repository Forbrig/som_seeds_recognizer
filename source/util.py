import numpy as np

# read the file dataset given and normalize all features to ranges between 0 an 1
def leitura_entrada(path):
    file = open(path)
    data = file.readlines()
    data = [x.replace('\n', '').replace('\t', ' ').replace('  ', ' ') for x in data]
    _features = np.array([line.split(' ')[0:8] for line in data], dtype = float)
    _class = np.array(_features[:,7]) # copy class column
    _features = np.delete(_features, 7, 1) # delete class column

    min = []
    max = []
    for i in range(0, 7):
        aux_min = np.amin(_features, axis = 0)[i]
        aux_max = np.amax(_features, axis = 0)[i]
        min.append(aux_min)
        max.append(aux_max)

    # normalize function x = (x - x_min)/(x_max-x_min)
    for i in _features:
        for j in range(0, 7):
            i[j] = (i[j] - min[j]) / (max[j] - min[j])
    #print(_features)

    return ([_features, _class])


'''
Função de distância euclidiana entre dois pontos.
'''
def dist_eclidiana(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(1/2)

'''
Função que retorna o item mais frequente em uma lista.
'''
def most_common(lst):
    return max(set(lst), key = lst.count)
