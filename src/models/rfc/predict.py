class PredictRFC():
    pass

    def predict_rfc(
            self,
            rfc, 
            X):
        
        return rfc.predict(X), rfc.predict_proba(X)
