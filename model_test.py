from keras.models import model_from_json
import numpy as np
import pandas
from sklearn.preprocessing import LabelEncoder

json_file = open('model_5_classes.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

loaded_model.load_weights("model_5_classes.h5")
print("Loaded model from disk")

loaded_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

dataframe_train = pandas.read_csv("5_classes.csv", header=None)
dataset_train = dataframe_train.values
names_train = dataset_train[:,0]
X_train = dataset_train[:,1:101].astype(float)
Y_train = dataset_train[:,101]

dataframe_test = pandas.read_csv("5_classes_validation.csv", header=None)
dataset_test = dataframe_test.values
names_test = dataset_test[:,0]
X_test = dataset_test[:,1:101].astype(float)
Y_test = dataset_test[:,101]



encoder = LabelEncoder()
encoder.classes_ = np.load('encoder_5.npy')
ynew_train = loaded_model.predict_classes(X_train)
ynew_train = encoder.inverse_transform(ynew_train)

ynew_test = loaded_model.predict_classes(X_test)
ynew_test = encoder.inverse_transform(ynew_test)

for i in range(len(X_train)):
    if Y_train[i] != ynew_train[i]:
        print("%d. Name -> %s   Correct=%s, Predicted=%s" %(i, names_train[i], Y_train[i], ynew_train[i]))

for i in range(len(X_test)):
    if Y_test[i] != ynew_test[i]:
        print("%d. Name -> %s   Correct=%s, Predicted=%s" %(i, names_test[i], Y_test[i], ynew_test[i]))