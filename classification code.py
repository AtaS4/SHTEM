import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

class MatClassRSA:
    class Preprocessing:
        @staticmethod
        def shuffleData(X, Y):
            idx = np.random.permutation(len(Y))
            return X[:, :, idx], Y[idx]

    class Classification:
        @staticmethod
        def crossValidateMulti(X, Y, channels=[96, 32]):
            # Prepare the data
            feature_vector = np.concatenate([X[ch_num, :, :] for ch_num in channels], axis=0).T
            X_train, X_test, Y_train, Y_test = train_test_split(feature_vector, Y, test_size=0.3)

            # Encode labels to one-hot vectors
            encoder = LabelEncoder()
            Y_train = encoder.fit_transform(Y_train)
            Y_test = encoder.transform(Y_test)
            Y_train = to_categorical(Y_train)
            Y_test = to_categorical(Y_test)

            # Build neural network model
            model = Sequential()
            model.add(Dense(512, input_dim=feature_vector.shape[1], activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(256, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(128, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))
            model.add(Dense(Y_train.shape[1], activation='softmax'))

            model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

            # Train the model with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            model.fit(X_train, Y_train, epochs=100, batch_size=64, validation_data=(X_test, Y_test), callbacks=[early_stopping])

            # Evaluate the model
            Y_pred = np.argmax(model.predict(X_test), axis=1)
            Y_test_labels = np.argmax(Y_test, axis=1)
            accuracy = accuracy_score(Y_test_labels, Y_pred)

            return accuracy

def load_data(pNum):
    X, labels6, labels72 = [], [], []
    if type(pNum) == int:
        pNum = [pNum]
    for i in pNum:
        baseFn = f'S{i:02d}.mat'
        data = sio.loadmat("./Data/" + baseFn)
        X.append(data['X'])
        labels6.append(data['labels6'])
        labels72.append(data['labels72'])
    X = np.concatenate(X, axis=2)
    labels6 = np.concatenate(labels6)
    labels72 = np.concatenate(labels72)
    return X, labels6, labels72, data['t'], data['fs']

def main(pNum, nCategories=6, channels=None):
    if channels is None:
        channels = range(124)  # Testing all 124 channels
    X, labels6, labels72, t, fs = load_data(pNum)
    RSA = MatClassRSA()
    Y = labels6 if nCategories == 6 else labels72
    Xr, Yr = RSA.Preprocessing.shuffleData(X, Y)

    accuracies = []
    for ch in channels:
        print(f'~ ~ Performing classification on channel {ch} ~ ~ ...')
        accuracy = RSA.Classification.crossValidateMulti(Xr, Yr.ravel(), [ch])
        accuracies.append(accuracy)

    # Plotting accuracy for each channel
    plt.figure()
    plt.plot(channels, accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Channel')
    plt.ylabel('Accuracy')
    plt.title('Channel vs Accuracy')
    plt.grid(True)
    plt.savefig(f'./Figures/S{pNum:02d}_accuracy_per_channel.png')
    plt.show()

if __name__ == "__main__":
    pNum = 1
    nCategories = 6
    main(pNum, nCategories, range(124))  # Test all 124 channels
