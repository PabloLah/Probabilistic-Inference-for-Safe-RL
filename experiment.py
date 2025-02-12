import logging

import hydra
import jax
from omegaconf import OmegaConf

from ssac.rl.trainer import get_state_path, load_state, should_resume, start_fresh

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

#P: teomporary ---disable/enable rendering---
import os
os.environ['MUJOCO_GL'] = 'egl'
#-------------------------------------
_LOG = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="ssac/configs", config_name="config")
def main(cfg):
    from ssac.agent.testing_fixed_classifier import CustomMLPClassifier
    _LOG.info(
        f"Setting up experiment with the following configuration: "
        f"\n{OmegaConf.to_yaml(cfg)}"
    )
    state_path = get_state_path()
    if should_resume(state_path):
        _LOG.info(f"Resuming experiment from: {state_path}")
        trainer = load_state(cfg, state_path)
    else:
        _LOG.info("Starting a new experiment.")
        trainer = start_fresh(cfg)
    with trainer, jax.disable_jit(not cfg.jit):
        trainer.train()
    _LOG.info("Done training.")


if __name__ == "__main__":
    main()
