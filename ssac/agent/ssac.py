import equinox as eqx
import jax
import numpy as np
from gymnasium.spaces import Box, Discrete
from omegaconf import DictConfig

from ssac.agent import safe_actor_critic as sac
from ssac.agent.replay_buffer import ReplayBuffer
from ssac.rl.epoch_summary import EpochSummary
from ssac.rl.metrics import MetricsMonitor
from ssac.rl.trajectory import Transition
from ssac.rl.types import FloatArray, Report
from ssac.rl.utils import PRNGSequence

from ssac.agent.testing_fixed_classifier import CustomMLPClassifier

@eqx.filter_jit
def policy(actor, observation, key):
    act = lambda o, k: actor.act(o, k)
    return eqx.filter_vmap(act)(
        observation, jax.random.split(key, observation.shape[0])
    )


class SafeSAC:
    def __init__(
        self,
        observation_space: Box | Discrete,
        action_space: Box | Discrete,
        config: DictConfig,
    ):
        self.prng = PRNGSequence(config.training.seed)
        self.config = config
        self.action_space = action_space
        self.replay_buffer = ReplayBuffer(
            config.agent.replay_buffer.capacity,
            config.training.seed,
            config.agent.batch_size,
        )
        self.actor_critic = sac.ActorCritic(
            observation_space, action_space, config, next(self.prng)
        )
        self.metrics_monitor = MetricsMonitor()
        self.classifier_max_wait = config.agent.classifier_retrain
        self.classifier_retrain_counter = self.classifier_max_wait + 1

    def __call__(self, observation: FloatArray, train: bool = True) -> FloatArray:
        
        if len(self.replay_buffer) > self.config.agent.prefill:
            #re-train classifier on all available data every config.agent.classifier_retrain times (loop)
            if self.classifier_retrain_counter > self.classifier_max_wait:
                self.actor_critic.reset_classifier(next(self.prng))
                for batch in self.replay_buffer.sample_all_batches():
                    metrics = self.actor_critic.retrain_classifier(batch, next(self.prng))
                    log(metrics, self.metrics_monitor)
                self.classifier_retrain_counter = 0
            #update actor
            for batch in self.replay_buffer.sample(self.config.training.parallel_envs):
                self.classifier_retrain_counter += 1 #count how many times actor is updated
                losses = self.actor_critic.update(batch, next(self.prng))
                log(losses, self.metrics_monitor)
            self.actor_critic.polyak(self.config.agent.polyak_rate, self.config.agent.safety_polyak_rate)
        action = policy(self.actor_critic.actor, observation, next(self.prng))
        return np.asarray(action)

    def observe(self, transition: Transition) -> None:
        self.replay_buffer.store(transition)

    def report(self, summary: EpochSummary, epoch: int, step: int) -> Report:
        metrics = {
            k: float(v.result.mean) for k, v in self.metrics_monitor.metrics.items()
        }
        self.metrics_monitor.reset()
        return Report(metrics=metrics)


def log(log_items, monitor):
    for k, v in log_items.items():
        monitor[k] = v.item()
