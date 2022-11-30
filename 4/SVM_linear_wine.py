'''

    Program rysuje wykresy z zestawieniami danych, porównując je do siebie na wykresie
    dane kolorowane są na podstawie tego w jakich klasach dane się znajdują

    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: requirements.txt

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki z zrzutami działającej aplikacji
    znajdują się w folderze: screenshots

'''

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_wine
import pandas as pd

# Load the dataset
wine = load_wine()

input_file = 'Decision_Tree/winequality-white.csv'

wine_csv_txt = np.loadtxt(input_file, skiprows=1, delimiter=';')

wine_csv = {}

wine_csv['data'] = wine_csv_txt[:, :11]
#
wine_csv['feature_names'] = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

wine_csv['target_names'] = ['Quality - 0', 'Quality - 1', 'Quality - 2', 'Quality - 3', 'Quality - 4', 'Quality - 5', 'Quality - 6', 'Quality - 7', 'Quality - 8', 'Quality - 9', 'Quality - 10']

wine_csv['target'] = wine_csv_txt[:, -1]

mylist = wine_csv['target']
mylist = list(dict.fromkeys(mylist))
mylist.sort()

#removing unexisting Qualities for this dataset
real_target = []
for item in mylist:
    for target in wine_csv['target_names']:
        if target.__contains__(str(int(item))):
            real_target.append(target)

# Printing all plots
for i in range(0, 10):
    plot = plt.scatter(wine_csv['data'][:, i], wine_csv['data'][:, i+1], c=wine_csv['target'])
    plt.title('Wine quality dataset')
    plt.xlabel(wine_csv['feature_names'][i])
    plt.ylabel(wine_csv['feature_names'][i+1])
    plt.legend(handles=plot.legend_elements()[0], labels=real_target,
                        loc="upper right", title="Quality class")
    plt.show()
