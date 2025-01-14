import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

COLORS = np.asarray(["black", "rosybrown", "chocolate", "darkorange", "yellow", "palegreen", "darkcyan", "slategray", "midnightblue", "purple"])


def svd_calculate(logits):
    cov_features = np.cov(np.transpose(logits))
    _, s, _ = np.linalg.svd(cov_features)
    log_s = np.log(s)
    return log_s

def svd_calculate_list(list_logits):
    n = len(list_logits)
    list_log_s = []
    for i in range(n):
        list_log_s.append(svd_calculate(list_logits[i]))
    return list_log_s

def svd_visulaize(list_logits, list_names, file_name, svd_name="SVD vis"):
    list_log_s = svd_calculate_list(list_logits)
    for i in range(len(list_log_s)):
        plt.plot(list_log_s[i], label=list_names[i])
    plt.legend()
    if file_name is None:
        file_name = "test.png"
    plt.savefig(file_name)
    plt.close()

if __name__ == '__main__':
    logits_1 = np.random.random((20, 10))
    logits_2 = np.random.random((20, 10))
    logits_3 = np.random.random((20, 10))
    list_logits_input = [logits_1, logits_2, logits_3]
    svd_visulaize(list_logits_input, ["1", "2", "3"], None)