'''

    Ten plik zajmuje się wizualizacja danych dla różnych wowołań tej metody


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

def visualize_classifier(classifier, X, y, title='', entrances=0):
    # Define the minimum and maximum values for X and Y
    # that will be used in the mesh grid

    # Define the step size to use in plotting the mesh grid 
    mesh_step_size = 0.01

    x_vals = []
    y_vals = []

    output = []

    # Define the mesh grid of X and Y values
    if entrances == 2:
        min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

        x_vals, y_vals = np.meshgrid(
            np.arange(min_y, max_y, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

        # Run the classifier on the mesh grid
        output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

        # Reshape the output array
        output = output.reshape(x_vals.shape)

        # Create a plot
        plt.figure()

        # Specify the title
        plt.title(title)

        # Choose a color scheme for the plot
        plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

        # Overlay the training points on the plot
        plt.scatter(X[:, 0], X[:, 1], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

        # Specify the boundaries of the plot
        plt.xlim(x_vals.min(), x_vals.max())
        plt.ylim(y_vals.min(), y_vals.max())

        # Specify the ticks on the X and Y axes
        plt.xticks((np.arange(int(X[:, 0].min() + 10), int(X[:, 0].max() - 10), 1.0)))
        plt.yticks((np.arange(int(X[:, 1].min() + 10), int(X[:, 1].max() - 10), 1.0)))

        plt.show()

    if entrances == 11:
        min_y, max_y = X[:, 9].min() - 10.0, X[:, 9].max() + 10.0

        min_1, max_1 = X[:, 0].min() - 10.0, X[:, 0].max() + 10.0
        min_2, max_2 = X[:, 1].min() - 10.0, X[:, 1].max() + 10.0
        min_3, max_3 = X[:, 2].min() - 10.0, X[:, 2].max() + 10.0
        min_4, max_4 = X[:, 3].min() - 10.0, X[:, 3].max() + 10.0
        min_5, max_5 = X[:, 4].min() - 10.0, X[:, 4].max() + 10.0
        min_6, max_6 = X[:, 5].min() - 10.0, X[:, 5].max() + 10.0
        min_7, max_7 = X[:, 6].min() - 10.0, X[:, 6].max() + 10.0
        min_8, max_8 = X[:, 7].min() - 10.0, X[:, 7].max() + 10.0
        min_9, max_9 = X[:, 8].min() - 10.0, X[:, 8].max() + 10.0
        min_10, max_10 = X[:, 9].min() - 10.0, X[:, 9].max() + 10.0
        min_11, max_11 = X[:, 10].min() - 10.0, X[:, 10].max() + 10.0

        mesh_step_size = 10000000

        vals_1, vals_2, vals_3, vals_4, vals_5, vals_6, vals_7, vals_8, vals_9, vals_10, vals_11 = np.meshgrid(
            np.arange(min_1, max_1, mesh_step_size),
            np.arange(min_2, max_2, mesh_step_size),
            np.arange(min_3, max_3, mesh_step_size),
            np.arange(min_4, max_4, mesh_step_size),
            np.arange(min_5, max_5, mesh_step_size),
            np.arange(min_6, max_6, mesh_step_size),
            np.arange(min_7, max_7, mesh_step_size),
            np.arange(min_8, max_8, mesh_step_size),
            np.arange(min_9, max_9, mesh_step_size),
            np.arange(min_10, max_10, mesh_step_size),
            np.arange(min_11, max_11, mesh_step_size)
        )

        x_vals, y_vals = np.meshgrid(
            np.arange(min_y, max_y, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

        # Run the classifier on the mesh grid
        output = classifier.predict(np.c_[vals_1.ravel(), vals_2.ravel(), vals_3.ravel(), vals_4.ravel(), vals_5.ravel(), vals_6.ravel(), vals_7.ravel(), vals_8.ravel(), vals_9.ravel(), vals_10.ravel(), vals_11.ravel()])

        print("output")
        print(output)
        # Reshape the output array
        output = output.reshape(x_vals.shape)

        # Create a plot
        plt.figure()

        # Specify the title
        plt.title(title)

        # Choose a color scheme for the plot
        plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

        # Overlay the training points on the plot
        plt.scatter(X[:, 8], X[:, 9], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

        # Specify the boundaries of the plot
        plt.xlim(x_vals.min(), x_vals.max())
        plt.ylim(y_vals.min(), y_vals.max())

        # Specify the ticks on the X and Y axes
        plt.xticks((np.arange(int(X[:, 8].min() + 2), int(X[:, 8].max() + 50), 40.0)))
        plt.yticks((np.arange(int(X[:, 9].min() + 2), int(X[:, 9].max() + 50), 40.0)))

        plt.show()


    if entrances == 13:

        min_x, max_x = X[:, 8].min() - 10.0, X[:, 8].max() + 10.0
        min_y, max_y = X[:, 9].min() - 10.0, X[:, 9].max() + 10.0

        min_1, max_1 = X[:, 0].min() - 10.0, X[:, 0].max() + 10.0
        min_2, max_2 = X[:, 1].min() - 10.0, X[:, 1].max() + 10.0
        min_3, max_3 = X[:, 2].min() - 10.0, X[:, 2].max() + 10.0
        min_4, max_4 = X[:, 3].min() - 10.0, X[:, 3].max() + 10.0
        min_5, max_5 = X[:, 4].min() - 10.0, X[:, 4].max() + 10.0
        min_6, max_6 = X[:, 5].min() - 10.0, X[:, 5].max() + 10.0
        min_7, max_7 = X[:, 6].min() - 10.0, X[:, 6].max() + 10.0
        min_8, max_8 = X[:, 7].min() - 10.0, X[:, 7].max() + 10.0
        min_9, max_9 = X[:, 8].min() - 10.0, X[:, 8].max() + 10.0
        min_10, max_10 = X[:, 9].min() - 10.0, X[:, 9].max() + 10.0
        min_11, max_11 = X[:, 10].min() - 10.0, X[:, 10].max() + 10.0
        min_12, max_12 = X[:, 11].min() - 10.0, X[:, 11].max() + 10.0
        min_13, max_13 = X[:, 12].min() - 10.0, X[:, 12].max() + 10.0

        mesh_step_size = 1000000

        vals_1, vals_2, vals_3, vals_4, vals_5, vals_6, vals_7, vals_8, vals_9, vals_10, vals_11, vals_12, vals_13 = np.meshgrid(
            np.arange(min_1, max_1, mesh_step_size),
            np.arange(min_2, max_2, mesh_step_size),
            np.arange(min_3, max_3, mesh_step_size),
            np.arange(min_4, max_4, mesh_step_size),
            np.arange(min_5, max_5, mesh_step_size),
            np.arange(min_6, max_6, mesh_step_size),
            np.arange(min_7, max_7, mesh_step_size),
            np.arange(min_8, max_8, mesh_step_size),
            np.arange(min_9, max_9, mesh_step_size),
            np.arange(min_10, max_10, mesh_step_size),
            np.arange(min_11, max_11, mesh_step_size),
            np.arange(min_12, max_12, mesh_step_size),
            np.arange(min_13, max_13, mesh_step_size)
        )

        x_vals, y_vals = np.meshgrid(
            np.arange(min_y, max_y, mesh_step_size), np.arange(min_y, max_y, mesh_step_size))

        # Run the classifier on the mesh grid
        output = classifier.predict(np.c_[vals_1.ravel(), vals_2.ravel(), vals_3.ravel(), vals_4.ravel(), vals_5.ravel(), vals_6.ravel(), vals_7.ravel(), vals_8.ravel(), vals_9.ravel(), vals_10.ravel(), vals_11.ravel(), vals_12.ravel(), vals_13.ravel()])

        #output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

        # Reshape the output array
        output = output.reshape(x_vals.shape)

        # Create a plot
        plt.figure()

        # Specify the title
        plt.title(title)

        # Choose a color scheme for the plot
        plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

        # Overlay the training points on the plot
        plt.scatter(X[:, 8], X[:, 9], c=y, s=75, edgecolors='black', linewidth=1, cmap=plt.cm.Paired)

        # Specify the boundaries of the plot
        plt.xlim(x_vals.min(), x_vals.max())
        plt.ylim(y_vals.min(), y_vals.max())

        # Specify the ticks on the X and Y axes
        plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 100), 1.0)))
        plt.yticks((np.arange(int(X[:, 4].min() - 1), int(X[:, 4].max() + 100), 1.0)))

        plt.show()





