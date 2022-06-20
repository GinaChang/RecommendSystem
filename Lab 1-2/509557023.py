from cProfile import label
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from matplotlib import pyplot as plt

df = pd.read_csv('glass.data.csv', header=None)
# print(df)
all_columns = ["Id number", "RI", "Na", "Mg", "Al",
               "Si", "K", "Ca", "Ba", "Fe", "Type of glass"]

df.columns = all_columns
df.drop("Id number", axis=1, inplace=True)
# print(df)

# Q1
main_features = all_columns[1:-1]
x = df[main_features]
y = df["Type of glass"]
# print(x)
# print(y)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=1, stratify=y)
print(f'x_train shape:{x_train.shape}')
print(f'y_train shape:{y_train.shape}')
print(f'x_test shape:{x_test.shape}')
print(f'y_test shape:{y_test.shape}')


# Q2 & Q3
types = [1, 2, 3, 5, 6, 7]
types_name = ['building-float', 'building-non-float',
              'vehicle-float', 'containers', 'tableware', 'headlamps']
model = RandomForestClassifier(n_estimators=25)
# GridSearchCV
# param_grid = {'n_estimators':np.arange(1,200)}
# model = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# train the model
model.fit(x_train, y_train)
# get GridSearch parameters
# best_n = model.best_params_['n_estimators']
# acc = model.best_score_
# print(f'Best n_estimators\t: {best_n}')

pred = model.predict(x_test)
# print prediction
# print(pred)
# use sklearn metrics
acc = accuracy_score(y_test, pred)
print(f'RandomForestClassifier Accuracy: {(acc*100).round(2)}%')
# print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred, labels=types)
# print(cm)
df_cm = pd.DataFrame(cm, index=types_name, columns=types_name)
# print(df_cm)


model = KNeighborsClassifier(n_neighbors=4)
# GridSearchCV
# param_grid = {'n_neighbors':np.arange(1,20)}
# model = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# train the model
model.fit(x_train, y_train)
# get GridSearch parameters
# best_n = model.best_params_['n_neighbors']
# acc = model.best_score_
# print(f'Best n_neighbors\t: {best_n}')

pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
print(f'KNeighborsClassifier Accuracy: {(acc*100).round(2)}%')
# print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred, labels=types)
# print(cm)
df_cm = pd.DataFrame(cm, index=types_name, columns=types_name)
# print(df_cm)


model = DecisionTreeClassifier()
# GridSearchCV
# param_grid = {'criterion': ['gini', 'entropy'], 'splitter': ['best', 'random']}
# model = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
# train the model
model.fit(x_train, y_train)
# get GridSearch parameters
# best_criterion = model.best_params_['criterion']
# best_splitter = model.best_params_['splitter']
# acc = model.best_score_
# print(f'Best criterion\t: {best_criterion}')
# print(f'Best splitter\t: {best_splitter}')

pred = model.predict(x_test)
acc = accuracy_score(y_test, pred)
print(f'DecisionTreeClassifier Accuracy: {(acc*100).round(2)}%')
# print(classification_report(y_test, pred))
cm = confusion_matrix(y_test, pred, labels=types)
# print(cm)
df_cm = pd.DataFrame(cm, index=types_name, columns=types_name)
# print(df_cm)
