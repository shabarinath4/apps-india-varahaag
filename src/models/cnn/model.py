import tensorflow as tf
from tensorflow.keras import layers, models

class TrainCNN:
    def __init__(self, classes, epochs, batch_size):
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size

    def network(self, shape):
        model = models.Sequential()
        model.add(layers.Conv1D(16, (3), activation="relu", input_shape=shape))
        model.add(layers.MaxPooling1D((1)))
        model.add(layers.Conv1D(32, (3), activation="relu"))
        model.add(layers.MaxPooling1D((1)))
        model.add(layers.Conv1D(64, (3), activation="relu"))
        model.add(layers.MaxPooling1D((1)))
        model.add(layers.Flatten())
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.classes, activation="softmax"))
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    
    def cnn(self, X_train, y_train, X_val=None, y_val=None):
        shape = X_train.shape[1:]
        model = self.network(shape)
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None else None,
                  verbose=2)
        return model