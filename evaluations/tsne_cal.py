import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


COLORS = np.asarray(["black", "rosybrown", "chocolate", "darkorange", "yellow", "palegreen", "darkcyan", "slategray", "midnightblue", "purple"])


def tsne_calculator(x, n_comp=2):
    return TSNE(n_components=n_comp, learning_rate='auto', init='random', perplexity=3).fit_transform(x)


def tsne_visualization(x, y, file_name="test.png", name="TSNE"):
    if name is None:
        name = "TSNE"
    colors = COLORS[y]
    plt.scatter(x[:,0], x[:, 1], c=colors, s=30)
    plt.title(name)
    plt.xlabel("X axis")
    plt.ylabel("Y axis")
    plt.savefig(file_name)


def logits_labels_tsne(x, y, file_name=None, name=None):
    x_2d = tsne_calculator(x)
    if file_name is not None:
        tsne_visualization(x_2d, y, file_name, name)
    return x_2d, y

if __name__ == '__main__':
    x = np.arange(5, 10)
    y = np.arange(12, 17)
    c = np.random.randint(0, 10, (5,))
    con_xy = np.stack((x, y))
    print(con_xy)
    tsne_visualization(con_xy, c)
