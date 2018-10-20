import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from keras.layers import Dropout

seed = 7
numpy.random.seed(seed)


dataframe = pandas.read_csv("cs_all_5_classes_tt.csv", header=None)
dataset = dataframe.values
X = dataset[:,0:100].astype(float)
Y = dataset[:,100]


encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)
dummy_y = np_utils.to_categorical(encoded_Y)


def first_model():
    def baseline_model():
        model = Sequential()
        model.add(Dense(50, input_dim=100, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    with open("models_5_classes.txt", "a") as myfile:
        myfile.write("Baseline: %.2f%% (%.2f%%)\n" % (results.mean()*100, results.std()*100))


def second_model():
    def baseline_model():
        model = Sequential()
        model.add(Dense(30, input_dim=100, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    with open("models_5_classes.txt", "a") as myfile:
        myfile.write("Baseline: %.2f%% (%.2f%%)\n" % (results.mean() * 100, results.std() * 100))


def third_model():
    def baseline_model():
        model = Sequential()
        model.add(Dense(30, input_dim=100, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    with open("models_5_classes.txt", "a") as myfile:
        myfile.write("Baseline: %.2f%% (%.2f%%)\n" % (results.mean() * 100, results.std() * 100))


def fourth_model():
    def baseline_model():
        model = Sequential()
        model.add(Dense(30, input_dim=100, activation='relu'))
        model.add(Dense(30, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    with open("models_5_classes.txt", "a") as myfile:
        myfile.write("Baseline: %.2f%% (%.2f%%)\n" % (results.mean() * 100, results.std() * 100))


def fifth_model():
    def baseline_model():
        model = Sequential()
        model.add(Dense(50, input_dim=100, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    with open("models_5_classes.txt", "a") as myfile:
        myfile.write("Baseline: %.2f%% (%.2f%%)\n" % (results.mean() * 100, results.std() * 100))


def sixth_model():
    def baseline_model():
        model = Sequential()
        model.add(Dense(20, input_dim=100, activation='relu'))
        model.add(Dense(5, activation='softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=1, verbose=0)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)

    results = cross_val_score(estimator, X, dummy_y, cv=kfold)

    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))
    with open("models_5_classes.txt", "a") as myfile:
        myfile.write("Baseline: %.2f%% (%.2f%%)\n" % (results.mean() * 100, results.std() * 100))

