import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer, f1_score

f1_macro_scorer = make_scorer(f1_score, average='macro')
    
class TrainRFC(object):

    def __init__(self):
        pass
    
    def rfc_cv(self, X, y):
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

    def rfc(self, X_train, y_train):
        
        rfc = self.rfc_cv(X_train, y_train)
        return rfc