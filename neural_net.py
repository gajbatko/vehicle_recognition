import numpy
import pandas
from keras.layers import Dropout
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle



seed = 7
numpy.random.seed(seed)


dataframe = pandas.read_csv("5_classes.csv", header=None)
dataset = dataframe.values
X_train = dataset[:,1:101].astype(float)
Y_train = dataset[:,101]
X_train, Y_train = shuffle(X_train, Y_train, random_state=7)


encoder = LabelEncoder()
encoder.fit(Y_train)
encoded_Y = encoder.transform(Y_train)
dummy_y = np_utils.to_categorical(encoded_Y)
numpy.save('encoder_5_3.npy', encoder.classes_)


model = Sequential()
model.add(Dense(30, input_dim=100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(20, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, dummy_y, epochs=200, batch_size=1, verbose=0)

model_json = model.to_json()
with open("model_5_3_classes.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model_5_3_classes.h5")

