from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import  GaussianNB
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

class StHelper:

    def __init__(self,X,y):
        self.X = X
        self.y = y
        # Apply train test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def create_base_estimators(self,estimators,voting_type):

        algos = []

        if 'KNN' in estimators:
            knn_clf = KNeighborsClassifier()
            algos.append(('knn', knn_clf))
        if 'Logistic Regression' in estimators:
            log_clf = LogisticRegression(solver="lbfgs", random_state=42)
            algos.append(('lr', log_clf))
        if 'Gaussian Naive Bayes' in estimators:
            gnb_clf = GaussianNB()
            algos.append(('gnb', gnb_clf))
        if 'SVM' in estimators:
            if voting_type == "hard":
                svm_clf = SVC(gamma="scale", random_state=42)
            else:
                svm_clf = SVC(gamma="scale", probability=True, random_state=42)
            algos.append(('svc', svm_clf))
        if 'Random Forest' in estimators:
            rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)
            algos.append(('rf', rnd_clf))

        return algos

    def train_voting_classifier(self,algos, voting_type):

        voting_clf = VotingClassifier(
            estimators=algos,
            voting=voting_type)

        voting_clf.fit(self.X_train, self.y_train)
        y_pred = voting_clf.predict(self.X_test)

        accuracy = accuracy_score(self.y_test, y_pred)

        return voting_clf, accuracy

    def draw_main_graph(self,voting_clf,ax):

        XX, YY, input_array = self.draw_meshgrid()
        labels = voting_clf.predict(input_array)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')

    def plot_other_graphs(self,algos):

        figs = []
        XX, YY, input_array = self.draw_meshgrid()

        for estimator in algos:
            estimator[1].fit(self.X_train, self.y_train)
            labels = estimator[1].predict(input_array)
            fig1, ax1 = plt.subplots()
            ax1.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
            figs.append(fig1)

        return figs

    def calculate_base_model_accuracy(self,algos):

        accuracy_scores = []

        for model in algos:
            model[1].fit(self.X_train, self.y_train)
            y_pred = model[1].predict(self.X_test)
            accuracy_scores.append(accuracy_score(self.y_test, y_pred))

        return accuracy_scores

    def draw_meshgrid(self):
        a = np.arange(start=self.X[:, 0].min() - 1, stop=self.X[:, 0].max() + 1, step=0.01)
        b = np.arange(start=self.X[:, 1].min() - 1, stop=self.X[:, 1].max() + 1, step=0.01)

        XX, YY = np.meshgrid(a, b)

        input_array = np.array([XX.ravel(), YY.ravel()]).T

        return XX, YY, input_array

        labels = voting_clf.predict(input_array)

