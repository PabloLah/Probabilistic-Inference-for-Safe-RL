#-----------------------------TESTING-----------------------------
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier

class CustomMLPClassifier:
    def __init__(self, state_dim, action_dim, hidden_size, max_iter=200, random_state=42):
        self.model = SklearnMLPClassifier(
            hidden_layer_sizes=(hidden_size,hidden_size),
            max_iter=max_iter,
            batch_size=256,
            tol=0.0000000000001,
            activation='relu',
            solver='adam',
            random_state=random_state,
            verbose=True
        )

    def train(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)[:, 1]  # Probability of class 1 (safe)
    
    def get_model_weights(self):
        return self.model.coefs_

    def set_model_weights(self, weights):
        self.model.coefs_ = [weights[key] for key in sorted(weights.keys())]
#------------------------------------------------------------------
