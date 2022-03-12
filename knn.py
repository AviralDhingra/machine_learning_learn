import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
#print(data.head())

# Converting all qualatative data into integers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))
 
predict = "class"  #optional

X = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = 0.1)
#print(x_train, x_test, y_train, y_test)

k = 7 # Number of neighbors
model = KNeighborsClassifier(n_neighbors=k)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)


predicted = model.predict(x_test)

"""
a = model.kneighbors_graph(X, k)
print(type(a))
"""



names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print(f'Data: {x_test[x]}\nPredicted: {predicted[x]}, Actual: {y_test[x]}')
    if predicted[x] == y_test[x]:
        print('Correct')
    else:
        print('Wrong')
    print("\n")
    
print(f'Final Accuracy: {round(acc*100)}%')
