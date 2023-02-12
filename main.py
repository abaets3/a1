import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import time

from sklearn import datasets, svm, metrics
from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_validate, KFold
from sklearn.metrics import accuracy_score

from nn import WineNetwork, PumpkinNetwork, train, get_predictions

#### GETTING THE DATA ####

# The wine quality dataset
wine = pd.read_csv('data/wine.csv')
wine = wine.drop('Id', axis=1)

wine_y = wine[["quality"]].copy()
wine_x = wine.drop("quality", axis=1)

wine_x_train, wine_x_test, wine_y_train, wine_y_test = train_test_split(wine_x,wine_y,test_size=.10,random_state=0)

# The pumpkin seeds dataset
pumpkin = pd.read_csv('data/pumpkin.csv')

pumpkin_y = pumpkin[["Class"]].copy()
pumpkin_x = pumpkin.drop("Class", axis=1)

pumpkin_x_train, pumpkin_x_test, pumpkin_y_train, pumpkin_y_test = train_test_split(pumpkin_x,pumpkin_y,test_size=.10,random_state=0)

#### HELPERS ####

def print_report(data_set, clf, predicted, actual):
    print(
        f"Classification report for classifier {clf} on {data_set}:\n"
        f"{metrics.classification_report(actual, predicted)}\n"
    )

    disp = metrics.ConfusionMatrixDisplay.from_predictions(actual, predicted)
    disp.figure_.suptitle(f"Confusion Matrix for {clf} on {data_set}")

    plt.savefig(f"plots/Confusion Matrix for {clf} on {data_set}.png")
    plt.close()


def create_subsample_plots(name, clf1, clf2):
    
    wine_test_acc = []
    wine_train_acc = []
    wine_train_time = []

    pumpkin_test_acc = []
    pumpkin_train_acc = []
    pumpkin_train_time = []
    for i in [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]:
        if i == 1:
            wine_x_train_sub, wine_y_train_sub = wine_x_train, wine_y_train
            pumpkin_x_train_sub, pumpkin_y_train_sub = pumpkin_x_train, pumpkin_y_train
        else:
            wine_x_train_sub, _, wine_y_train_sub, _ = train_test_split(wine_x_train,wine_y_train,test_size=1-i,random_state=0)
            pumpkin_x_train_sub, _, pumpkin_y_train_sub, _ = train_test_split(pumpkin_x_train,pumpkin_y_train,test_size=1-i,random_state=0)
        
        start = time.time()
        clf1.fit(wine_x_train_sub, wine_y_train_sub)
        wine_train_time.append((time.time() - start))
        predicted = clf1.predict(wine_x_test)
        wine_test_acc.append(accuracy_score(wine_y_test, predicted))
        predicted = clf1.predict(wine_x_train_sub)
        wine_train_acc.append(accuracy_score(wine_y_train_sub, predicted))

        start = time.time()
        clf2.fit(pumpkin_x_train_sub, pumpkin_y_train_sub)
        pumpkin_train_time.append((time.time() - start))
        predicted = clf2.predict(pumpkin_x_test)
        pumpkin_test_acc.append(accuracy_score(pumpkin_y_test, predicted))
        predicted = clf2.predict(pumpkin_x_train_sub)
        pumpkin_train_acc.append(accuracy_score(pumpkin_y_train_sub, predicted))

    plt.plot([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], wine_train_acc, label="Wine Quality Train Accuracy")
    plt.plot([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], wine_test_acc, label="Wine Quality Test Accuracy")
    plt.plot([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], pumpkin_train_acc, label="Pumpkin Quality Train Accuracy")
    plt.plot([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], pumpkin_test_acc, label="Pumpkin Quality Test Accuracy")
    plt.xlabel("Percentage of training data used")
    plt.ylabel("Accuracy")
    plt.title(f"Affect of Sampling Training Data on Accuracy on {name}")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/Affect of Sampling Training Data on Accuracy on {name}")
    plt.close()


    plt.plot([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], wine_train_time, label="Wine Quality Train Time")
    plt.plot([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1], pumpkin_train_time, label="Pumpkin Quality Train Time")
    plt.xlabel("Percentage of training data used")
    plt.ylabel("Time (seconds)")
    plt.title(f"Affect of Sampling Training Data on Time on {name}")
    plt.legend(loc="upper left")
    plt.savefig(f"plots/Affect of Sampling Training Data on Time on {name}")
    plt.close()

#### DECISION TREES ####

def decision_trees():
    wine_val_avg_f1 = []
    wine_train_avg_f1 = []
    for i in range(1,50):
        clf = DecisionTreeClassifier(max_depth=i)
        scores = cross_validate(clf, wine_x_train, wine_y_train, cv=5, scoring='f1_macro', return_train_score=True)
        wine_val_avg_f1.append(scores['test_score'].mean())
        wine_train_avg_f1.append(scores['train_score'].mean())
    plt.plot(wine_val_avg_f1, label="Mean Validation F1 Macro Score")
    plt.plot(wine_train_avg_f1, label="Mean Training F1 Macro Score")
    plt.xlabel("Decision Tree Max Depth")
    plt.ylabel("Mean F1 Macro Scores")
    plt.title("Decision Tree Max Depth versus F1 Scores")
    plt.legend(loc="lower right")
    plt.savefig("plots/Decision Tree Max Depth on Wine")
    plt.close()

    pumpkin_val_avg_f1 = []
    pumpkin_train_avg_f1 = []
    for i in range(1,50):
        clf = DecisionTreeClassifier(max_depth=i)
        scores = cross_validate(clf, pumpkin_x_train, pumpkin_y_train, cv=5, scoring='f1_macro', return_train_score=True)
        pumpkin_val_avg_f1.append(scores['test_score'].mean())
        pumpkin_train_avg_f1.append(scores['train_score'].mean())
    plt.plot(pumpkin_val_avg_f1, label="Mean Validation F1 Macro Score")
    plt.plot(pumpkin_train_avg_f1, label="Mean Training F1 Macro Score")
    plt.xlabel("Decision Tree Max Depth")
    plt.ylabel("Mean F1 Macro Scores")
    plt.title("Decision Tree Max Depth versus F1 Macro Scores")
    plt.legend(loc="lower right")
    plt.savefig("plots/Decision Tree Max Depth on Pumpkin")
    plt.close()    

#### NEURAL NETWORKS ####

def neural_networks():
    for j in [.001, .01, .1]:
        kf = KFold(5)
        avg_training_loss = []
        avg_validation_loss = []
        for i, (train_index, val_index) in enumerate(kf.split(wine_x_train)):
            model = WineNetwork()
            model, training_loss, validation_loss = train(
                model, wine_x_train.values[train_index], wine_y_train.values[train_index], wine_x_train.values[val_index], wine_y_train.values[val_index], 
                128, j, 0, 150)
            avg_training_loss.append(training_loss)
            avg_validation_loss.append(validation_loss)
        np.mean(avg_training_loss, axis=0)
        np.mean(avg_validation_loss, axis=0)
        plt.plot(training_loss, label=f"Training Loss for learning rate of {j}")
        plt.plot(validation_loss, label=f"Validation Loss for learning rate of {j}")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Validation Loss")
    plt.ylim([.006, .018])
    plt.title("Affect of Learning Rate on Cross Validation Loss")
    plt.savefig("plots/Learning Rate impact on Wine")
    plt.close()


    for j in [0, .001, .01]:
        kf = KFold(5)
        avg_training_loss = []
        avg_validation_loss = []
        for i, (train_index, val_index) in enumerate(kf.split(wine_x_train)):
            model = WineNetwork()
            model, training_loss, validation_loss = train(
                model, wine_x_train.values[train_index], wine_y_train.values[train_index], wine_x_train.values[val_index], wine_y_train.values[val_index], 
                128, .01, j, 150)
            avg_training_loss.append(training_loss)
            avg_validation_loss.append(validation_loss)
        np.mean(avg_training_loss, axis=0)
        np.mean(avg_validation_loss, axis=0)
        plt.plot(training_loss, label=f"Training Loss for weight decay of {j}")
        plt.plot(validation_loss, label=f"Validation Loss for weight decay of {j}")
    plt.legend(loc="upper right")
    plt.xlabel("Epochs")
    plt.ylabel("Mean Validation Loss")
    plt.title("Affect of Weight Decay on Cross Validation Loss")
    plt.savefig("plots/Weight Decay impact on Wine")
    plt.close()

    model = WineNetwork()
    model, training_loss, validation_loss = train(model, wine_x_train.values, wine_y_train.values, wine_x_test.values, wine_y_test.values, 128, .001, 0, 100)
    predicted = get_predictions(model, wine_x_test)
    print_report("Wine Quality", "Neural Net", predicted, wine_y_test)

    clf = MLPClassifier().fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "Neural Net", predicted, pumpkin_y_test)
    
#### BOOSTED DECISION TREES ####

def boosted():
    for i in range(5, 25, 5):
        wine_val_avg_f1 = []
        for j in range(1, 100):
            dt = DecisionTreeClassifier(max_depth=i)
            clf = AdaBoostClassifier(dt, n_estimators=j)
            scores = cross_validate(clf, wine_x_train, np.ravel(wine_y_train), cv=5, scoring='f1_macro')
            wine_val_avg_f1.append(scores['test_score'].mean())
        plt.plot(wine_val_avg_f1, label=f"Mean Cross Validation F1 Macro Score with max_depth of {i}")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("Mean Validation F1 Macro Scores")
    plt.title("AdaBoost Decision Tree Learners versus F1 Scores on Wine Quality")
    plt.legend(loc="lower right")
    plt.savefig("plots/Boosted Decision Learners on Wine")
    plt.close()

    for i in range(1, 4):
        pumpkin_val_avg_f1 = []
        for j in range(1, 50):
            dt = DecisionTreeClassifier(max_depth=i)
            clf = AdaBoostClassifier(dt, n_estimators=j)
            scores = cross_validate(clf, pumpkin_x_train, np.ravel(pumpkin_y_train), cv=5, scoring='f1_macro')
            pumpkin_val_avg_f1.append(scores['test_score'].mean())
        plt.plot(pumpkin_val_avg_f1, label=f"Mean Cross Validation F1 Macro Score with max_depth of {i}")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("Mean Validation F1 Macro Scores")
    plt.title("AdaBoost Decision Tree Learners versus F1 Scores on Pumpkin Seeds")
    plt.legend(loc="lower right")
    plt.savefig("plots/Boosted Decision Learners on Pumpkin")
    plt.close()


    pumpkin_val_avg_f1 = []
    pumpkin_train_avg_f1 = []
    for j in range(1, 50):
        dt = DecisionTreeClassifier(max_depth=1)
        clf = AdaBoostClassifier(dt, n_estimators=j)
        scores = cross_validate(clf, pumpkin_x_train, np.ravel(pumpkin_y_train), cv=5, scoring='f1_macro', return_train_score=True)
        pumpkin_val_avg_f1.append(scores['test_score'].mean())
        pumpkin_train_avg_f1.append(scores['train_score'].mean())
    plt.plot(pumpkin_train_avg_f1, label="Mean Training F1 Macro Score")
    plt.plot(pumpkin_val_avg_f1, label="Mean Validation F1 Macro Score")
    plt.xlabel("Number of Weak Learners")
    plt.ylabel("Mean Validation F1 Macro Scores")
    plt.title("AdaBoost Decision Tree Learners versus F1 Scores on Pumpkin Seeds")
    plt.legend(loc="lower right")
    plt.savefig("plots/Overfitting Boosted Decision Learners on Pumpkin")
    plt.close()

    dt = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(dt, n_estimators=12)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "AdaBoost Decision Tree", predicted, pumpkin_y_test)

    dt = DecisionTreeClassifier(max_depth=5)
    clf = AdaBoostClassifier(dt, n_estimators=100)
    clf.fit(wine_x_train, wine_y_train)
    predicted = clf.predict(wine_x_test)
    print_report("Wine Quality", "AdaBoost Decision Tree", predicted, wine_y_test)

#### SUPPORT VECTOR MACHINES ####

def support():

    for kernel in ["rbf", "sigmoid"]:
        pumpkin_val_avg_f1 = []
        wine_val_avg_f1 = []
        for i in [.00001, .0001, .001, .01, .1, 1, 10]:
            clf = svm.SVC(kernel=kernel, gamma=i)
            scores = cross_validate(clf, pumpkin_x_train, np.ravel(pumpkin_y_train), cv=5, scoring='f1_macro')
            pumpkin_val_avg_f1.append(scores['test_score'].mean())

            clf = svm.SVC(kernel=kernel, gamma=i)
            scores = cross_validate(clf, wine_x_train, np.ravel(wine_y_train), cv=5, scoring='f1_macro')
            wine_val_avg_f1.append(scores['test_score'].mean())

        plt.plot([.00001, .0001, .001, .01, .1,  1, 10], pumpkin_val_avg_f1, label=f"Pumpkin Seeds Dataset on {kernel}")
        plt.plot([.00001, .0001, .001, .01, .1,  1, 10], wine_val_avg_f1, label=f"Wine Quality Dataset on {kernel}")

    plt.xscale("log")
    plt.xlabel("Gamma")
    plt.ylabel("Mean Validation F1 Macro Scores")
    plt.title("Impact of Gamma on Kernels")
    plt.legend(loc="upper right")
    plt.savefig("plots/Impact of Gamma on Kernels")
    plt.close()

    for kernel in ["rbf", "poly", "linear", "sigmoid"]:
        if kernel == "rbf":
            clf = svm.SVC(kernel=kernel, gamma=.0001)
        else:
            clf = svm.SVC(kernel=kernel)
        scores = cross_validate(clf, pumpkin_x_train, np.ravel(pumpkin_y_train), cv=5, scoring='f1_macro')
        print(f"Performance on cross val on SVC using {kernel} kernel was: {(scores['test_score'].mean())}")

        if kernel == "rbf":
            clf = svm.SVC(kernel=kernel, gamma=.0001)
        else:
            clf = svm.SVC(kernel=kernel)
        clf.fit(pumpkin_x_train, np.ravel(pumpkin_y_train))
        predicted = clf.predict(pumpkin_x_test)
        print_report("Pumpkin Seeds", f"SVM with {kernel} kernel", predicted, pumpkin_y_test)

    
    for kernel in ["rbf", "poly", "linear", "sigmoid"]:
        if kernel == "rbf":
            clf = svm.SVC(kernel=kernel, gamma=1)
        else:
            clf = svm.SVC(kernel=kernel)
        scores = cross_validate(clf, wine_x_train, np.ravel(wine_y_train), cv=5, scoring='f1_macro')
        print(f"Performance on cross val on SVC using {kernel} kernel was: {(scores['test_score'].mean())}")

        if kernel == "rbf":
            clf = svm.SVC(kernel=kernel, gamma=1)
        else:
            clf = svm.SVC(kernel=kernel)
        clf.fit(wine_x_train, np.ravel(wine_y_train))
        predicted = clf.predict(wine_x_test)
        print_report("Wine Quality", f"SVM with {kernel} kernel", predicted, wine_y_test)

#### KNNs ####

def knns():

    wine_val_avg_f1 = []
    wine_train_avg_f1 = []
    for i in range(1,50):
        clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_validate(clf, wine_x_train, np.ravel(wine_y_train), cv=5, scoring='f1_macro', return_train_score=True)
        wine_val_avg_f1.append(scores['test_score'].mean())
        wine_train_avg_f1.append(scores['train_score'].mean())
    plt.plot(wine_val_avg_f1, label="Mean Validation F1 Macro Score")
    plt.plot(wine_train_avg_f1, label="Mean Training F1 Macro Score")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Mean F1 Macro Scores")
    plt.title("Number of neighbors versus F1 Scores")
    plt.legend(loc="upper right")
    plt.savefig("plots/KNN impact of n on Wine")
    plt.close()

    pumpkin_val_avg_f1 = []
    pumpkin_train_avg_f1 = []
    for i in range(1,50):
        clf = KNeighborsClassifier(n_neighbors=i)
        scores = cross_validate(clf, pumpkin_x_train, np.ravel(pumpkin_y_train), cv=5, scoring='f1_macro', return_train_score=True)
        pumpkin_val_avg_f1.append(scores['test_score'].mean())
        pumpkin_train_avg_f1.append(scores['train_score'].mean())
    plt.plot(pumpkin_val_avg_f1, label="Mean Validation F1 Macro Score")
    plt.plot(pumpkin_train_avg_f1, label="Mean Training F1 Macro Score")
    plt.xlabel("Number of neighbors")
    plt.ylabel("Mean F1 Macro Scores")
    plt.title("Number of neighbors versus F1 Scores")
    plt.legend(loc="upper right")
    plt.savefig("plots/KNN impact of n on Pumpkin")
    plt.close()

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "KNN", predicted, pumpkin_y_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(wine_x_train, wine_y_train)
    predicted = clf.predict(wine_x_test)
    print_report("Wine Quality", "KNN", predicted, wine_y_test)

def winners():
    clf = DecisionTreeClassifier(max_depth=4)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "Decision Tree", predicted, pumpkin_y_test)

    clf = DecisionTreeClassifier(max_depth=24)
    clf.fit(wine_x_train, wine_y_train)
    predicted = clf.predict(wine_x_test)
    print_report("Wine Quality", "Decision Tree", predicted, wine_y_test)

    dt = DecisionTreeClassifier(max_depth=1)
    clf = AdaBoostClassifier(dt, n_estimators=10)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "AdaBoost Decision Tree", predicted, pumpkin_y_test)

    dt = DecisionTreeClassifier(max_depth=8)
    clf = AdaBoostClassifier(dt, n_estimators=25)
    clf.fit(wine_x_train, wine_y_train)
    predicted = clf.predict(wine_x_test)
    print_report("Wine Quality", "AdaBoost Decision Tree", predicted, wine_y_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "KNN", predicted, pumpkin_y_test)

    clf = KNeighborsClassifier(n_neighbors=1)
    clf.fit(wine_x_train, wine_y_train)
    predicted = clf.predict(wine_x_test)
    print_report("Wine Quality", "KNN", predicted, wine_y_test)

    model = WineNetwork()
    model, training_loss, validation_loss = train(model, wine_x_train.values, wine_y_train.values, wine_x_test.values, wine_y_test.values, 128, .01, .005, 100)
    predicted = get_predictions(model, wine_x_test)
    print_report("Wine Quality", "Neural Net", predicted, wine_y_test)

    clf = MLPClassifier().fit(pumpkin_x_train, pumpkin_y_train)
    predicted = clf.predict(pumpkin_x_test)
    print_report("Pumpkin Seeds", "Neural Net", predicted, pumpkin_y_test)

neural_networks()
decision_trees()
support()
knns()
boosted()
winners()
create_subsample_plots("Decision Trees", DecisionTreeClassifier(max_depth=24), DecisionTreeClassifier(max_depth=4))
create_subsample_plots("Boosted Decision Trees", AdaBoostClassifier(DecisionTreeClassifier(max_depth=24), n_estimators=25), AdaBoostClassifier(DecisionTreeClassifier(max_depth=4), n_estimators=10))
create_subsample_plots("Neural Networks", MLPClassifier(), MLPClassifier())
create_subsample_plots("KNN", KNeighborsClassifier(n_neighbors=1), KNeighborsClassifier(n_neighbors=1))
create_subsample_plots("SVC", svm.SVC(kernel="linear"), svm.SVC(kernel="linear"))
