import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier

'''

    opis zadan w pliku: opis_zadan.txt

   Program jest przerobionym poprzednim zadaniem z wine quality dataset, z dodanymi sieciami neuronowymi
    gdzie porównujemy klasyfikacje data_decision_tree_wine.py starej wersji z nową zawierającą sieci neuronowe

    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: requirements.txt

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki z zrzutami działającej aplikacji
    znajdują się w folderze: screenshots_NAI5

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load input data
input_file = 'winequality-white.csv'

wine_csv_txt = np.loadtxt(input_file, skiprows=1, delimiter=';')

wine_csv = {}
wine_csv['data'] = wine_csv_txt[:, :12]
wine_csv['feature_names'] = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]
wine_csv['target_names'] = ['Quality - 0', 'Quality - 1', 'Quality - 2', 'Quality - 3', 'Quality - 4', 'Quality - 5', 'Quality - 6', 'Quality - 7', 'Quality - 8', 'Quality - 9', 'Quality - 10']
wine_csv['target'] = wine_csv_txt[:, -1]

# Split data into training and testing datasets
x_train, x_test, y_train, y_test = train_test_split(
    wine_csv['data'], wine_csv['target'], test_size=0.25, random_state=5)

# Decision Trees classifier
params = {'random_state': 5, 'max_depth': 8}

classifier = DecisionTreeClassifier(**params)
classifier.fit(x_train, y_train)
y_test_pred = classifier.predict(x_test)

mylist = y_train
mylist = list(dict.fromkeys(mylist))
mylist.sort()

real_target = []
for item in mylist:
    for target in wine_csv['target_names']:
        if target.__contains__(str(int(item))):
            real_target.append(target)

print("\n" + "#" * 40)
print("\nClassifier performance on training dataset\n")
print(classification_report(y_train, classifier.predict(x_train), target_names=real_target))
print("#" * 40 + "\n")

mylist_test = y_test
mylist_test = list(dict.fromkeys(mylist_test))
mylist_test.sort()

real_target = []
for item in mylist_test:
    for target in wine_csv['target_names']:
        if target.__contains__(str(int(item))):
            real_target.append(target)

print("existing target names")
print(real_target)

print("#" * 40)
print("\nClassifier performance on test dataset\n")
print(classification_report(y_test, y_test_pred, target_names=real_target))
print("#" * 40 + "\n")

plt.show()

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