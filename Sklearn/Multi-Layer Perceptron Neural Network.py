from sklearn.neural_network import MLPClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_diabetes
from sklearn.neural_network import MLPRegressor

X, y = load_breast_cancer(return_X_y=True)

print(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=2)

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

clf.score(X_test, y_test)



A, b = load_diabetes(return_X_y=True)
A_train, A_test, b_train, b_test = train_test_split(A, b,test_size=0.2,
                                                    random_state=2)
regr = MLPRegressor(random_state=1, max_iter=500).fit(A_train, b_train)

print(A)

regr.score(A_test, b_test)
