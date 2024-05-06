import pickle
from sklearn import datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
labels = {
    0: "setosa",
    1: "versicolor",
    2: "virginica"
}

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.60)

from sklearn import tree
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)

pickle.dump(classifier, open('model.pkl', 'wb'))

model = pickle.load(open('model.pkl', 'rb'))

x = [[6.7, 3.3, 5.7, 2.1]]
predict = model.predict(x)
print(predict)
print("Hello world")
print(labels[predict[0]])