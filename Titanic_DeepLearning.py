
#?     Library importing

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from keras.models import Sequential
from keras import layers
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import graphviz
from math import sqrt
from sklearn.tree import export_graphviz
from seaborn import regplot

#?--------------------------

#* Reading titanic data and preparing it for machine learning form

titanic_data = pd.read_csv('titanic.csv')
titanic_data = titanic_data.drop(['Name', 'Ticket', 'Cabin', 'Embarked'], axis=1)
titanic_data = pd.get_dummies(titanic_data, columns=['Sex', 'Pclass'], drop_first=True)
titanic_data = titanic_data.dropna()

titanic_attirs = titanic_data.drop('Survived', axis=1)
titanic_value = titanic_data['Survived']

#* Splitting data to test and train
x_train,x_test,y_train,y_test = train_test_split(titanic_attirs,titanic_value, test_size=0.2,random_state=42)


#* Creating Neural Network model
model_lr = Sequential([
    layers.Input(shape=(x_train.shape[1],)),
    layers.Dense(32, activation="tanh"),
    layers.Dense(1, activation="sigmoid")
])


optimizer = tf.keras.optimizers.Adam()
loss = tf.keras.losses.BinaryCrossentropy()

#* Compiling NN model
model_lr.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
model_lr.summary()

#* Training NN model
history_lr = model_lr.fit(x_train, y_train, epochs=50, batch_size=16, validation_data=(x_test, y_test))


#? Details about NN model
x= model_lr.evaluate(x_test,y_test)

print("NN model score: " +str(x[1]))
#! OUTPUT ~= 0.80

y_test = np.array(y_test)
y_pred = model_lr.predict(x_test)

mse = mean_squared_error(y_test,y_pred)
rmse = sqrt(mse)

average_of_actual = np.average(y_test)

ssr = 0
for i in range(0,len(y_test)):
    ssr += (y_pred[i] - average_of_actual)**2
ssr = ssr[0]

ssto = 0
for x in range(0,len(y_test)):
    ssto += (y_test[x] - average_of_actual)**2

sse = 0
for x in range(0,len(y_test)):
    sse += (y_test[x] -y_pred[x])**2
sse = sse[0]

r = sqrt(1-(sse/ssto))
r_square = r**2

print("mse: "+str(mse))
print("rmse: "+str(rmse))
print("ssr: "+str(ssr))
print("sse: "+str(sse))
print("ssto: "+str(ssto))
print("r square: "+str(r_square))

#? Details end

#* Plotting NN model

plt.plot(history_lr.history["loss"], label= "train")
plt.plot(history_lr.history["val_loss"], label= "val")
plt.ylabel("loss")
plt.legend()
plt.show()

plt.plot(history_lr.history["accuracy"], label= "train")
plt.plot(history_lr.history["val_accuracy"], label= "val")
plt.ylabel("accuracy")
plt.legend()
plt.show()

#? Random Forest model

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
forest = rf.fit(x_train, y_train)

tree = forest.estimators_[5]
forest.score(x_test,y_test)
feature_list = ["Age","Fare","Parch","Pclass_2","Pclass_3","Sex_male","SibSp"]

#* Drawing decision tree
for i in range(1):
    tree = rf.estimators_[i]
    dot_data = export_graphviz(tree,
                               feature_names=x_train.columns,  
                               filled=True, 
                               impurity=False, 
                               proportion=True)
    graph = graphviz.Source(dot_data)

x= forest.score(x_test,y_test)
print("RandomForest score: " +str(x))
#! OUTPUT ~= 0.32

#? Logistic Regression Model

Lr = LogisticRegression()
lr_model = Lr.fit(x_train,y_train)
lr_model.score(x_test,y_test)
y_pred = lr_model.predict(x_test)

#*Plotting logistic regression
regplot(x=x_test["Parch"],y=y_pred, data=titanic_data,logistic=True)

print("Logistic Regression Model Score: "+str(lr_model.score(x_test,y_test)))
#! OUTPUT ~= 0.76
