import pandas as pd

concentric = pd.read_csv('toy_datasets/concertriccir2.csv')
linear = pd.read_csv('toy_datasets/linearsep.csv')
outlier = pd.read_csv('toy_datasets/outlier.csv')
spiral = pd.read_csv('toy_datasets/twoSpirals.csv')
ushape = pd.read_csv('toy_datasets/ushape.csv')
xor = pd.read_csv('toy_datasets/xor.csv')

def load_dataset():

    return concentric,linear,outlier,spiral,ushape,xor

def load_initial_graph(dataset,ax):
    if dataset == "U-Shaped":
        ax.scatter(ushape['X'], ushape['Y'], c=ushape['class'], cmap='rainbow')
        df = ushape
    elif dataset == "Linearly Separable":
        ax.scatter(linear['X'], linear['Y'], c=linear['class'], cmap='rainbow')
        df = linear
    elif dataset == "Outlier":
        ax.scatter(outlier['X'], outlier['Y'], c=outlier['class'], cmap='rainbow')
        df = outlier
    elif dataset == "Two Spirals":
        ax.scatter(spiral['X'], spiral['Y'], c=spiral['class'], cmap='rainbow')
        df = spiral
    elif dataset == "Concentric Circles":
        ax.scatter(concentric['X'], concentric['Y'], c=concentric['class'], cmap='rainbow')
        df = concentric
    elif dataset == "XOR":
        ax.scatter(xor['X'], xor['Y'], c=xor['class'], cmap='rainbow')
        df = xor

    return df