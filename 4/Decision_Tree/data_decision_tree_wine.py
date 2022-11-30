import csv

'''

    Program rysuje wykresy z danymi treningowymi i testowymi
    oraz wyświetla w konsoli wydajność klasyfikatora na testowym i treningowym zbiorze danych
    jak i również rysuje wykres wejściowych danych


    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: requirements.txt

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki z zrzutami działającej aplikacji
    znajdują się w folderze: screenshots

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from utilities import visualize_classifier

# Load input data
input_file = 'winequality-white.csv'

wine_csv_txt = np.loadtxt(input_file, skiprows=1, delimiter=';')

wine_csv = {}

wine_csv['data'] = wine_csv_txt[:, :11]
#
wine_csv['feature_names'] = ["fixed acidity", "volatile acidity", "citric acid", "residual sugar", "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol", "quality"]

wine_csv['target_names'] = ['Quality - 0', 'Quality - 1', 'Quality - 2', 'Quality - 3', 'Quality - 4', 'Quality - 5', 'Quality - 6', 'Quality - 7', 'Quality - 8', 'Quality - 9', 'Quality - 10']

wine_csv['target'] = wine_csv_txt[:, -1]

# Visualize input data
plt.figure()

colors_set = np.array(["orange","purple","beige","brown","gray","cyan","magenta","red","green","blue","yellow","pink"])

for i in range(0, 11):
    probe = wine_csv['data'][wine_csv['target'] == float(i)]

    if len(probe) > 0:
        #print(" printuje i " + str(i) + " " + str(probe))
        plt.scatter(np.array(probe)[:, 0], np.array(probe)[:, 10], s=75, facecolors=colors_set[i],
                edgecolors=colors_set[i], linewidth=1, marker='o')

plt.title('Input data')


# Split data into training and testing datasets
X_train, X_test, y_train, y_test = train_test_split(
    wine_csv['data'], wine_csv['target'], test_size=0.25, random_state=5)

# Decision Trees classifier
params = {'random_state': 5, 'max_depth': 8}

classifier = DecisionTreeClassifier(**params)

classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Training dataset', 11)

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Test dataset', 11)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_test_pred)
print('Confusion matrix\n\n', cm)

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
print(classification_report(y_train, classifier.predict(X_train), target_names=real_target))
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
