import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix,accuracy_score, classification_report
from sklearn.metrics import make_scorer, f1_score
from sklearn.datasets import make_classification
import tensorflow as tf
from tensorflow.keras import layers, models

f1_macro_scorer = make_scorer(f1_score, average='macro')
    
class Baseline(object):

    def __init__(self):
        pass
    
    def rfc_cv(self, X, y):
        """
        Purpose:
        Parameters:
        Returns:
        """
        rfv = RandomForestClassifier(class_weight='balanced',oob_score=True)

        rf_params={'bootstrap': [True],
                    'max_depth': [100,120,140,160],
                    'n_estimators': [300,500,700,900,1100],
                    'min_samples_leaf': [1, 2],
                    'min_samples_split': [2, 5]}

        rf_random = RandomizedSearchCV(estimator = rfv,
                                    param_distributions = rf_params, n_iter =5,
                                    cv = 3, verbose=2, random_state=42, n_jobs = -1, scoring=f1_macro_scorer)

        model=rf_random.fit(X, y)

        return model.best_estimator_ 

    def baseline_rfc(self, X_train, y_train):
        
        rfc = self.rfc_cv(X_train, y_train)

        return rfc

class CNNBaseline:
    def __init__(self, input_shape, classes, epochs, batch_size):
        self.input_shape = input_shape
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size

    def network(self):
        model = models.Sequential()
        model.add(layers.Conv1D(16, (3), activation="relu", input_shape=self.input_shape))
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
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        model = self.network()
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None else None,
                  verbose=2)
        return model
    
class LSTMBaseline:
    def __init__(self, input_shape, classes, epochs, batch_size):
        self.input_shape = input_shape
        self.classes = classes
        self.epochs = epochs
        self.batch_size = batch_size

    def network(self):
        model = models.Sequential()
        model.add(layers.Masking(input_shape=self.input_shape))
        model.add(layers.LSTM(16, return_sequences=True))
        model.add(layers.LSTM(32, return_sequences=True))
        model.add(layers.LSTM(64, activation="relu"))
        model.add(layers.Dense(64, activation="relu"))
        model.add(layers.Dense(self.classes, activation="softmax"))
        
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        model = self.network()
        model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size,
                validation_data=(X_val, y_val) if X_val is not None else None,
                  verbose=2)
        return model