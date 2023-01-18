import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier
from sklearn.metrics import accuracy_score

def split_train_test(X, y, proportion_train):
    # proportion_train is a value between 0.1 and 1
    # determines the proportion of data used for training. the remaining data is kept for testing
    
    # X, y = l2.extract_features_labels()

    nb_data = len(X)
    nb_train = int(proportion_train * nb_data)
    nb_test = nb_data - nb_train

    Y = np.array([y, -(y - 1)]).T
    tr_X = X[:nb_train]
    tr_Y = Y[:nb_train]
    te_X = X[nb_train:]
    te_Y = Y[nb_train:]

    return tr_X, tr_Y, te_X, te_Y, nb_train, nb_test

def general_classifier(training_images, training_labels, test_images, test_labels, classifier):
    classifier.fit(training_images, training_labels)

    pred = classifier.predict(test_images)

    # print(pred)
    # print("Accuracy:", accuracy_score(test_labels, pred))
    return accuracy_score(test_labels, pred)

def multi_train_test(tr_X, tr_Y, te_X, te_Y, nb_train_data, nb_test_data):
    kernels = ["linear", "poly", "rbf", "sigmoid"]

    # contains models and their score
    classifiers = {
        "Random Forest": RandomForestClassifier(n_estimators=100)
    }

    for k in range(1, 10):
        classifiers[f"KNN (n={k})"] = KNeighborsClassifier(n_neighbors=k)

        classifiers[f"Bagging (n={k})"] = BaggingClassifier(n_estimators=k,max_samples=0.5, max_features=4,random_state=1)

    
    for kernel in kernels:
        classifiers[f"SVM ({kernel})"] = svm.SVC(kernel=kernel)

    
    sorted_models = sorted(classifiers.keys(), key=lambda x:x.lower())

    results = []
    for model in sorted_models:
        clf = classifiers[model]

        pred = general_classifier(tr_X.reshape((nb_train_data, 68*2)), list(zip(*tr_Y))[0], te_X.reshape((nb_test_data, 68*2)), list(zip(*te_Y))[0], clf)

        results.append(pred)
        # print(model, pred)

    return sorted_models, results
