import timeit
import sys
import functools as ft
import memory_profiler
from mlwpy import *
from sklearn import datasets, naive_bayes, neighbors, model_selection as skms

def knn_go(train_ftrs, test_ftrs, train_tgt):
    knn = neighbors.KNeighborsClassifier(n_neighbors=3)
    fit = knn.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)

def nb_go(train_ftrs, test_ftrs, train_tgt):
    nb = naive_bayes.GaussianNB()
    fit = nb.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)

def split_data(dataset):
    split = skms.train_test_split(dataset.data, dataset.target, test_size=0.25)
    return split[:-1]  # drop test target

def msr_time(go, args):
    call = ft.partial(go, *args)
    tu = min(timeit.Timer(call).repeat(repeat=3, number=100))
    print("{:<6}: ~{:.4f} sec".format(go.__name__, tu))

def msr_mem(go, args):
    base = memory_profiler.memory_usage()[0]
    mu = memory_profiler.memory_usage((go, args), max_usage=True)
    print("{:<6}: ~{:.4f} MiB".format(go.__name__, mu - base))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("No arguments provided. Using defaults: time knn")
        which_msr = 'time'
        which_go = 'knn'
    else:
        which_msr = sys.argv[1]  # 'time' or 'mem'
        which_go = sys.argv[2]   # 'knn' or 'nb'

    msr = {'time': msr_time, 'mem': msr_mem}[which_msr]
    go = {'nb': nb_go, 'knn': knn_go}[which_go]
    sd = split_data(datasets.load_iris())
    msr(go, sd)
