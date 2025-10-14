class PredictCNN():
    pass

    def predict_cnn(
            self, 
            cnn, 
            X):
        
        y_probs = cnn.predict(X)
        return y_probs.argmax(axis=1), y_probs