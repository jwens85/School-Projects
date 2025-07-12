import timeit
import sys
import functools as ft
import memory_profiler
from mlwpy import *
from sklearn import datasets, neighbors, linear_model, model_selection as skms

def knn_go(train_ftrs, test_ftrs, train_tgt):
    knn = neighbors.KNeighborsRegressor(n_neighbors=3)
    fit = knn.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)

def lr_go(train_ftrs, test_ftrs, train_tgt):
    linreg = linear_model.LinearRegression()
    fit = linreg.fit(train_ftrs, train_tgt)
    preds = fit.predict(test_ftrs)

def split_data(dataset):
    split = skms.train_test_split(dataset.data, dataset.target, test_size=0.25)
    return split[:-1]  # don't need test target

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
        print("No arguments provided. Using defaults: time lr")
        which_msr = 'time'
        which_go = 'lr'
    else:
        which_msr = sys.argv[1]  # 'time' or 'mem'
        which_go = sys.argv[2]   # 'lr' or 'knn'

    msr = {'time': msr_time, 'mem': msr_mem}[which_msr]
    go = {'lr': lr_go, 'knn': knn_go}[which_go]

    # Use a regression dataset
    diabetes = datasets.load_diabetes()
    sd = split_data(diabetes)

    msr(go, sd)
