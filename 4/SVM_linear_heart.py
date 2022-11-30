
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

input_file = 'Decision_Tree/heart_failure_clinical_records_dataset.csv'

wine_csv_txt = np.loadtxt(input_file, skiprows=1, delimiter=',')

wine_csv = {}

wine_csv['data'] = wine_csv_txt[:, :14]

wine_csv['feature_names'] = ['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time','DEATH_EVENT']

wine_csv['target_names'] = ['YES', 'NO']

wine_csv['target'] = wine_csv_txt[:, -1]

# Printing all plots
for i in range(0, 11):
    plot = plt.scatter(wine_csv['data'][:, i], wine_csv['data'][:, i+1], c=wine_csv['target'])
    plt.title('Do you have chance for heart failure?')
    plt.xlabel(wine_csv['feature_names'][i])
    plt.ylabel(wine_csv['feature_names'][i+1])
    plt.legend(handles=plot.legend_elements()[0], labels=wine_csv['target_names'],
                        loc="upper right", title="Quality class")
    plt.show()
