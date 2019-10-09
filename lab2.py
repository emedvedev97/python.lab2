import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz #, plot_tree
from sklearn.model_selection import GridSearchCV
#import matplotlib.pyplot as plt

x_train = pd.read_csv("/home/emedvedev/pythonLab/lab2/train.csv")
x_test = pd.read_csv("/home/emedvedev/pythonLab/lab2/test.csv")
y_test = pd.read_csv("/home/emedvedev/pythonLab/lab2/gender_submission.csv")

y_test = y_test['Survived']
y_train = x_train['Survived']

x_test = pd.concat([y_test,  x_test ],  axis=1)
x_train.drop(['PassengerId'], axis=1, inplace=True)
x_test.drop(['PassengerId'], axis=1, inplace=True)


x_train[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
x_train.drop(['Survived',  'Name', 'Ticket'], axis=1, inplace=True)
x_train.drop(['Cabin'], axis=1, inplace=True)
x_train['Embarked']
x_train['Age'].fillna(x_train['Age'].median(), inplace=True)
x_train['Embarked'].fillna('S', inplace=True)
x_train = pd.concat([x_train,  pd.get_dummies(x_train['Embarked'], prefix="Embarked")],  axis=1)
x_train.drop(['Embarked'], axis=1, inplace=True)
x_train['Sex'] = pd.factorize(x_train['Sex'])[0]


x_test[["Sex", "Survived"]].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived', ascending=False)
x_test.drop(['Survived',  'Name', 'Ticket'], axis=1, inplace=True)
x_test.drop(['Cabin'], axis=1, inplace=True)
x_test['Embarked']
x_test['Age'].fillna(x_test['Age'].median(), inplace=True)
x_test['Embarked'].fillna('S', inplace=True)
x_test = pd.concat([x_test,  pd.get_dummies(x_test['Embarked'], prefix="Embarked")],  axis=1)
x_test.drop(['Embarked'], axis=1, inplace=True)
x_test['Sex'] = pd.factorize(x_test['Sex'])[0]

x_test['Fare'].fillna(x_test['Fare'].median(), inplace=True)

param_grid = {'max_depth': [i for i in range(1,100)],
'max_leaf_nodes': [i for i in range(2,100)]}


grid = GridSearchCV(DecisionTreeClassifier(random_state=13), param_grid, cv=5)
grid.fit(x_train, y_train)
print("Правильность на тренеровачном наборе: {:.2f}".format(grid.score (x_train, y_train)))
print("Наилучшие значения параметров: {}".format(grid.best_params_))
print("Наилучшее значение кросс-валидац. правильности:{:.2f}".format(grid.best_score_))
print("Правильность на тестовом наборе: {:.2f}".format(grid.score(x_test, y_test)))
#plot_tree(grid.best_estimator_)
#plt.show()
export_graphviz(grid.best_estimator_, feature_names=x_train.columns.values, out_file='/home/emedvedev/pythonLab/lab2/tree.dot', filled=True)
