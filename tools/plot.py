import matplotlib.pyplot as plt
import numpy as np

def plot_data(x, y, precision="", plot_bests_only=False, print_results=False, save=False):
    if plot_bests_only:
        best_bagging = np.argmax(y[0:9])
        best_knn = np.argmax(y[9:18])
        best_rand_forest = 18
        best_svm = np.argmax(y[19:23])

        x = [x[best_bagging], x[best_knn + 9], x[18], x[best_svm + 19]]

        y = [y[best_bagging], y[best_knn + 9], y[18], y[best_svm + 19]]


    if print_results:
        for i in range(len(x)):
            print(x[i], y[i])
        
        
    fig, ax = plt.subplots()

    plt.title(f"Percentage of Succcess of Various Models ({precision})")
    plt.xticks(rotation='vertical')
    plt.bar(x, y)
    plt.gcf().subplots_adjust(bottom=0.25)

    plt.show()

    if save:
        fig.savefig(f"{precision}.png")
