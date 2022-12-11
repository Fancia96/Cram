import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

'''
    opis zadan w pliku: opis_zadan.txt

   Program uzywa sieci neuronowe i uczy je przewidywania pożarów w Algerii
   
    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: requirements.txt

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki z zrzutami działającej aplikacji
    znajdują się w folderze: screenshots_NAI5

'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load input data
input_file = 'Algerian_forest_fires_dataset_UPDATE.csv'

fire_csv_txt = np.loadtxt(input_file, skiprows=1, usecols=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), delimiter=',')

fire_csv = {}
fire_csv['data'] = fire_csv_txt[:, :13]
fire_csv['feature_names'] = ['day', 'month', 'year', 'Temperature', 'RH', 'Ws', 'Rain', 'FFMC', 'DMC', 'DC', 'ISI', 'BUI', 'FWI', 'Classes']
fire_csv['target_names'] = ['fire', 'not fire']
fire_csv['target'] = np.loadtxt(input_file, skiprows=1, dtype="U", usecols=13, delimiter=',')

for i in range(0, len(fire_csv['target'])):
    fire_csv['target'][i] = fire_csv['target'][i].strip()

fire_csv['target'].reshape(1, -1)

# Split data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(
    fire_csv['data'], fire_csv['target'], test_size=0.25, random_state=5)

y_test.reshape(-1, 1)

# Decision Trees classifier
params = {'random_state': 5, 'max_depth': 8}

classifier = DecisionTreeClassifier(**params)
classifier.fit(x_train, y_train)
y_test_pred = classifier.predict(x_test)

y_train = y_train.reshape(-1, 1)

###############################################

# set up MLP Classifier
mlp = MLPClassifier(
    hidden_layer_sizes=(50,),
    max_iter=15,
    alpha=1e-4,
    solver="sgd",
    verbose=True,
    random_state=1,
    learning_rate_init=0.1
)

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning, module="sklearn")

mlp.fit(x_train, y_train)

# print out the model scores
print(f"Training set score: {mlp.score(x_train, y_train)}")
print(f"Test set score: {mlp.score(x_test, y_test)}")
