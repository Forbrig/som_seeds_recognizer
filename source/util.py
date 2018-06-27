import matplotlib.pyplot as plt
import numpy as np

# read the file dataset given and normalize all features to ranges between 0 an 1
def read_file(path):
    file = open(path)
    data = file.readlines()
    data = [x.replace('\n', '').replace('\t', ' ').replace('  ', ' ') for x in data]
    _features = np.array([line.split(' ')[0:8] for line in data], dtype = float)
    np.random.shuffle(_features)
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

# plot neuron map, with colors
def plot_grid(grid, GRID_SIZE_I, GRID_SIZE_J):
    print(grid)
    kama_x = []
    kama_y = []
    rosa_x = []
    rosa_y = []
    cana_x = []
    cana_y = []
    none_x = []
    none_y = []
    for x in range(0, GRID_SIZE_I):
        for y in range(0, GRID_SIZE_J):
            if (x,y) in grid:
                if grid[(x,y)] == 1:
                    kama_x.append(x)
                    kama_y.append(y)
                elif grid[(x,y)] == 2:
                    rosa_x.append(x)
                    rosa_y.append(y)
                elif grid[(x,y)] == 3:
                    cana_x.append(x)
                    cana_y.append(y)
            else:
                none_x.append(x)
                none_y.append(y)
    plt.scatter(kama_x, kama_y, c = 'blue', label = 'Kama')
    plt.scatter(rosa_x, rosa_y, c = 'red', label = 'Rosa')
    plt.scatter(cana_x, cana_y, c = 'green', label = 'Canadian')
    plt.scatter(none_x, none_y, c = 'grey', label = 'None')
    plt.title('Neuron Grid')
    plt.legend(loc = 'upper center', bbox_to_anchor=(0.5, -0.05), ncol = 4)
    plt.show()

# euclidian distance
def dist_eclidiana(a, b):
    return ((a[0] - b[0])**2 + (a[1] - b[1])**2)**(1/2)

# most common item on a list
def most_common(lst):
    return max(set(lst), key = lst.count)
