from typing import NamedTuple, Optional

import distrax as trx
import equinox as eqx
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
from gymnasium.spaces import Box, Discrete
from gym.spaces import Box as gym_Box
from omegaconf import DictConfig
from optax import OptState, l2_loss, sigmoid_binary_cross_entropy

from ssac.rl.learner import Learner
from ssac.rl.trajectory import Transition
from ssac.agent.testing_fixed_classifier import CustomMLPClassifier





#-----------------------------TESTING-----------------------------
import sklearn
#jax.config.update("jax_numpy_rank_promotion", "raise")  # or "raise" or "warn"
#jax.config.update("jax_numpy_rank_promotion", False)
from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier

class CustomMLPClassifier:
    def __init__(self, state_dim, action_dim, hidden_size, max_iter=1, random_state=42):#dont train here
        self.model = SklearnMLPClassifier(
            hidden_layer_sizes=(hidden_size,hidden_size),
            max_iter=max_iter,
            batch_size=256,
            tol=0.0000000000001,
            activation='relu',
            solver='adam',
            random_state=random_state,
            verbose=False
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
    
class WrappedSklearnMLPClassifier:
    def __init__(self, model: CustomMLPClassifier):
        self.model = model

    def predict_proba(self, X):
        return self.model.predict_proba(X)


#------------------------------------------------------------------

def inv_softplus(x):
    return jnp.where(x < 20.0, jnp.log(jnp.exp(x) - 1.0), x)


class ActorBase(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(state_dim, action_dim, hidden_size, n_layers, key=key)

    def __call__(self, state: jax.Array) -> trx.Distribution:
        raise NotImplementedError

    def act(
        self,
        state: jax.Array,
        key: Optional[jax.Array] = None,
        deterministic: bool = False,
    ) -> jax.Array:
        if deterministic:
            return self(state).mean()
        else:
            assert key is not None
            return self(state).sample(seed=key)


class ContinuousActor(ActorBase):
    net: eqx.nn.MLP

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        n_layers: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(state_dim, action_dim * 2, hidden_size, n_layers, key=key)

    def __call__(self, state: jax.Array) -> trx.MultivariateNormalDiag:
        #P: adapt compatibility with safe_adaptation_gym; possibly not necessary
        #need to flatten the vector, since gym does not do this automatically
        if state.ndim > 1:
            state = state.flatten()
        #-----^
        x = self.net(state)
        mu, stddev = jnp.split(x, 2, axis=-1)
        stddev = jnn.softplus(stddev) + 1e-5
        dist = trx.Normal(mu, stddev)
        dist = trx.Transformed(dist, trx.Tanh())
        dist = trx.Independent(dist, 1)
        return dist


class DiscreteActor(ActorBase):
    net: eqx.nn.MLP

    def __call__(self, state: jax.Array) -> trx.Categorical:
        logits = self.net(state)
        # TODO (yarden): not sure if distrax's implementation uses
        # reparameterization/relaxation for taking gradients
        #  through samples of Categorical
        return trx.Categorical(logits=logits) 


class Critic(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(
            state_dim + (action_dim if action_dim is not None else 0),
            1,
            hidden_size,
            n_layers,
            key=key,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        x = self.net(jnp.concatenate([state, action], axis=-1))
        return x.squeeze(-1)


class CriticPair(NamedTuple):
    uno_critic: Critic
    dos_critic: Critic

    def __call__(
        self, state: jax.Array, action: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        return self.uno_critic(state, action), self.dos_critic(state, action)

#make safety classifier (NN to predict whether cost c<=0; probability)
class Classifier(eqx.Module):
    net: eqx.nn.MLP

    def __init__(
        self,
        n_layers: int,
        state_dim: int,
        action_dim: int,
        hidden_size: int,
        *,
        key: jax.Array,
    ):
        self.net = eqx.nn.MLP(
            state_dim + action_dim,
            1,
            hidden_size,
            n_layers,
            key=key,
        )

    def __call__(self, state: jax.Array, action: jax.Array) -> jax.Array:
        x = self.net(jnp.concatenate([state, action], axis=-1))
        #x = jax.nn.sigmoid(x) # let model predict logits instead of probabilities
        return x.squeeze(-1)
    
    def set_weights_from_custom_mlp(self, weights):
        # Get the current weights of the eqx MLP
        eqx_weights = eqx.tree_at(lambda mlp: mlp.weights, self.net)
        eqx_biases = eqx.tree_at(lambda mlp: mlp.biases, self.net)

        # Flatten the list of weights and biases from eqx MLP
        eqx_weights_flat = [w for layer_weights in eqx_weights for w in layer_weights]
        eqx_biases_flat = [b for layer_biases in eqx_biases for b in layer_biases]

        # Assign weights from CustomMLPClassifier to eqx MLP
        for i, w in enumerate(weights):
            eqx_weights_flat[i] = jnp.array(w)

        # Assign biases from CustomMLPClassifier to eqx MLP (assuming 0 biases)
        for i, b in enumerate([0]*len(eqx_biases_flat)):
            eqx_biases_flat[i] = jnp.array(b)

        # Reconstruct the eqx MLP weight and bias tree
        self.net = eqx.tree_at(lambda mlp: mlp.weights, self.net, eqx_weights_flat)
        self.net = eqx.tree_at(lambda mlp: mlp.biases, self.net, eqx_biases_flat)

def make_classifier(
    observation_space: Box | Discrete,
    action_space: Box | Discrete,
    config: DictConfig,
    key: jax.Array,
):
    state_dim = np.prod(observation_space.shape)  # type: ignore
    if isinstance(action_space, (Box, gym_Box)):
        action_dim = np.prod(action_space.shape)  # type: ignore
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
    else:
        raise NotImplementedError
    classifier = Classifier(
        state_dim=state_dim, action_dim=action_dim, **config.agent.classifier, key=key
    )
    return classifier

def make_actor(
    observation_space: Box | Discrete,
    action_space: Box | Discrete,
    config: DictConfig,
    key: jax.Array,
) -> ContinuousActor | DiscreteActor:
    state_dim = np.prod(observation_space.shape)  # type: ignore
    #check for Box entity from either gym.spaces or gymnasium.spaces;
    #P: adapted this in 3 parts of this code. (Box, gym_Box) instead of Box
    if isinstance(action_space, (Box, gym_Box)):
        action_dim = np.prod(action_space.shape)  # type: ignore
        actor = ContinuousActor(
            state_dim=state_dim, action_dim=action_dim, **config.agent.actor, key=key
        )
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
        actor = DiscreteActor(
            state_dim=state_dim, action_dim=action_dim, **config.agent.actor, key=key
        )
    else:
        raise NotImplementedError
    return actor


def make_critics(
    observation_space: Box | Discrete,
    action_space: Box | Discrete,
    config: DictConfig,
    key: jax.Array,
):
    state_dim = np.prod(observation_space.shape)  # type: ignore
    if isinstance(action_space, (Box, gym_Box)):
        action_dim = np.prod(action_space.shape)  # type: ignore
    elif isinstance(action_space, Discrete):
        action_dim = action_space.n
    else:
        raise NotImplementedError
    critic_factory = lambda key: Critic(
        state_dim=state_dim, action_dim=action_dim, **config.agent.critic, key=key
    )
    key1, key2 = jax.random.split(key)
    one = critic_factory(key1)
    two = critic_factory(key2)
    return CriticPair(one, two)


class ActorCritic:
    def __init__(
        self,
        observation_space: Box | Discrete,
        action_space: Box | Discrete,
        config: DictConfig,
        key: jax.Array
    ) -> None:
        classifier_key, actor_key, critic_key, target_key, safety_critic_key, safety_target_key = jax.random.split(key, 6)
        self.actor = make_actor(observation_space, action_space, config, actor_key)
        self.actor_learner = Learner(self.actor, config.agent.actor_optimizer)
        self.critics = make_critics(observation_space, action_space, config, critic_key)
        self.target_critics = make_critics(
            observation_space, action_space, config, target_key
        )
        self.critics_learner = Learner(self.critics, config.agent.critic_optimizer)

        #TAG: FIXED_CLASSIFIER
        '''self.classifier = make_classifier(observation_space, action_space, config, classifier_key)
        self.classifier_learner = Learner(self.classifier, config.agent.classifier_optimizer)
        #----------temporary -----------
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config
        self.classifier_key = classifier_key
        self.classifier_reset_counter = 0
        self.classifier_max_updates = config.agent.classifier_max_updates'''
        import joblib
        from ssac.agent.testing_fixed_classifier import CustomMLPClassifier
        from sklearn.neural_network import MLPClassifier as SklearnMLPClassifier
        #loaded_classifier = joblib.load('/cluster/home/plahmann/safe-control-as-inference/sk_classifier.pkl')
        #self.classifier = WrappedSklearnMLPClassifier(loaded_classifier)
        #values are hard coded for this environment

        #classifier = joblib.load('sk_classifier.pkl')

        #print(type(classifier.get_model_weights()))
        #print(classifier.get_model_weights()[0])
        #weights = classifier.get_model_weights()
        #np.savez('classifier_weights.npz', *weights)
        
        #hard coded values for environment safe-gym point to goal
        self.classifier = CustomMLPClassifier(60, 2, 64)
        # Generate dummy data for training
        np.random.seed(0)  # For reproducibility

        # Dummy features (X_train) and labels (y_train)
        X_train = np.random.rand(20, 62)  # 20 samples, 5 features
        y_train = np.random.randint(0, 2, size=20)  # Binary labels (0 or 1)
        self.classifier.train(X_train, y_train)


        loaded_weights = np.load('/cluster/home/plahmann/safe-control-as-inference/classifier_weights.npz', allow_pickle=True)
        self.classifier.set_model_weights(loaded_weights)

        '''#----second possible solution----
        self.classifier = make_classifier(observation_space, action_space, config, classifier_key)
        loaded_weights = np.load('/cluster/home/plahmann/safe-control-as-inference/classifier_weights.npz', allow_pickle=True)
        self.classifier.set_weights_from_custom_mlp(loaded_weights)'''

        #----------------------
        self.safety_critics = make_critics(observation_space, action_space, config, safety_critic_key)
        self.target_safety_critics = make_critics(
            observation_space, action_space, config, safety_target_key)

        self.safety_critics_learner = Learner(self.safety_critics, config.agent.safety_critic_optimizer)
        self.safety_discount = config.agent.safety_discount

        self.discount = config.agent.discount
        self.log_lagrangians = jnp.asarray(config.agent.initial_log_lagrangians)
        self.log_lagrangians_learner = Learner(
            self.log_lagrangians, config.agent.lagrangians_optimizer
        )
        self.target_entropy = np.prod(action_space.shape)  # type: ignore
        self.target_safety = config.training.safety_budget #NOT currently used

    def update(self, batch: Transition, key: jax.Array) -> dict[str, float]:
        classifier_key, actor_key, critics_key, safety_critics_key = jax.random.split(key, 4)
        lagrangians = jnp.exp(self.log_lagrangians)
        (self.critics, self.critics_learner.state), critic_loss = update_critics(
            batch,
            self.critics,
            self.target_critics,
            self.actor,
            self.critics_learner.state,
            self.critics_learner,
            lagrangians,
            self.discount,
            critics_key,
        )
        '''#----reset classifier------------- TAG: FIXED_CLASSIFIER
        self.classifier_reset_counter += 1
        if self.classifier_reset_counter % self.classifier_max_updates == 0:
            self.reset_classifier()
        #---------------------------------
        (self.classifier, self.classifier_learner.state), classifier_out = update_classifier(
            batch,
            self.classifier,
            self.classifier_learner.state,
            self.classifier_learner,
            classifier_key,
        )'''

        (self.safety_critics, self.safety_critics_learner.state), safety_critic_loss = update_safety_critics(
            batch,
            self.safety_critics,
            self.target_safety_critics,
            self.classifier,
            self.actor,
            self.safety_critics_learner.state,
            self.safety_critics_learner,
            lagrangians,
            self.safety_discount,
            safety_critics_key,
        )

        (self.actor, self.actor_learner.state), rest = update_actor(
            batch,
            self.actor,
            self.critics,
            self.safety_critics,
            self.actor_learner.state,
            self.actor_learner,
            lagrangians,
            actor_key,
        )
        (
            (self.log_lagrangians, self.log_lagrangians_learner.state),
            lagrangian_loss,
        ) = update_log_lagrangians(
            self.log_lagrangians,
            rest["log_pi"],
            jnp.asarray(self.target_entropy),
            self.log_lagrangians_learner.state,
            self.log_lagrangians_learner,
        )
        #into dict: TAG: FIXED CLASSIFIER
        '''classifier_loss=classifier_out["loss"],
            classifier_mean_prediction=classifier_out["preds_prob"],
            classification_mean_cost=classifier_out["costs"],
            logit_min=classifier_out["logit_min"], 
            logit_10_quantile=classifier_out["logit_10_quantile"], 
            logit_mean=classifier_out["logit_mean"],'''
        out = dict(
            critic_loss=critic_loss,
            safety_critic_loss=safety_critic_loss,
            lagrangian_loss=lagrangian_loss,
            actor_loss=rest["loss"],
            actor__critic_component=rest["critic_component"], 
            actor__safety_critic_component=rest["safety_critic_component"], 
            actor__surprise_component=rest["surprise_component"],
            lagrangian=jnp.exp(self.log_lagrangians.sum()),
        )
        return out

    def polyak(self, rate: float, safety_rate: float):
        only_arrays = lambda tree: eqx.filter(tree, eqx.is_array)
        updates = jax.tree_map(
            lambda a, b: rate * (a - b),
            only_arrays(self.critics),
            only_arrays(self.target_critics),
        )
        self.target_critics = eqx.apply_updates(self.target_critics, updates)

        #repeat for safety critics
        safety_updates = jax.tree_map(
            lambda a, b: safety_rate * (a - b),
            only_arrays(self.safety_critics),
            only_arrays(self.target_safety_critics),
        )
        self.target_safety_critics = eqx.apply_updates(self.target_safety_critics, safety_updates)
    
    def reset_classifier(self):
        #reset classifier to retrain form scratch; for more stable safety classification
        #todo: cleaner passing of variables
        self.classifier = make_classifier(self.observation_space, self.action_space, self.config, self.classifier_key)
        self.classifier_learner = Learner(self.classifier, self.config.agent.classifier_optimizer)
        self.classifier_reset_counter = 0


@eqx.filter_jit
def update_critics(
    batch: Transition,
    critics: CriticPair,
    target_critics: CriticPair,
    actor: ContinuousActor | DiscreteActor,
    learning_state: OptState,
    learner: Learner,
    lagrangians: jax.Array,
    discount: float,
    key: jax.Array,
) -> tuple[tuple[CriticPair, OptState], jax.Array]:
    terminals = jnp.zeros_like(batch.reward)

    def loss_fn(critics):
        #sample action, based on observation o
        sample_log_prob = jax.vmap(lambda o: actor(o).sample_and_log_prob(seed=key))
        next_action, log_pi = sample_log_prob(batch.next_observation)
        #compute target critic value based on observation and taken action (Q-val?)
        #does NOT update target network
        next_qs = jax.vmap(target_critics)(batch.next_observation, next_action)
        debiased_q = jnp.minimum(*next_qs)
        #compute log term of policy probability
        surprise = -lagrangians * log_pi
        next_soft_q = debiased_q + surprise
        soft_q_target = batch.reward + (1.0 - terminals) * discount * next_soft_q
        soft_q_target = jax.lax.stop_gradient(soft_q_target)
        #computes critic networks prediction. 
        q1, q2 = jax.vmap(critics)(batch.observation, batch.action)
        #compute loss for both critic networks
        return (l2_loss(q1, soft_q_target) + l2_loss(q2, soft_q_target)).mean()
    #do optimization step for critic network (NOT target critic)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(critics)
    new_critics, new_learning_state = learner.grad_step(critics, grads, learning_state)
    return (new_critics, new_learning_state), loss

#----------------------------------
#implement update classifier function 
@eqx.filter_jit
def update_classifier(
    batch: Transition,
    classifier: Classifier,
    learning_state: OptState,
    learner: Learner,
    key: jax.Array,
) -> tuple[tuple[Classifier, OptState], jax.Array]:    
    def loss_fn(classifier):
        # Predict probabilities
        preds = jax.vmap(classifier)(batch.observation, batch.action)

        # Transform costs into binary targets (1 for costs <= 0, 0 for costs > 0, i.e. probability of safety)
        binary_targets = (batch.cost <= 0).astype(jnp.float32)

        # Compute binary cross-entropy loss; need logits
        loss = sigmoid_binary_cross_entropy(preds, binary_targets)
        return loss.mean(), (jnp.mean(jax.nn.sigmoid(preds)), jnp.mean(batch.cost), jnp.min(preds), jnp.quantile(preds, 0.1), jnp.mean(preds)) #(preds, batch.cost) #(debugging)if we use the wandb logging in update function

    # Compute loss and gradients
    (loss, (preds_prob, costs, logit_min, logit_10_quantile, logit_mean)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(classifier)
    # Perform optimization step
    new_classifier, new_learning_state = learner.grad_step(classifier, grads, learning_state)
    return (new_classifier, new_learning_state), dict(loss=loss, preds_prob=preds_prob, costs=costs, logit_min=logit_min, logit_10_quantile=logit_10_quantile, logit_mean=logit_mean)

#----------------------------------
#ADD UPDATE SAFETY CRITICS FUNCTION; (mostly) ANALOGOUS TO update_critics
# delete unnecessary inputs
#@eqx.filter_jit
def update_safety_critics(
    batch: Transition,
    safety_critics: CriticPair,
    target_safety_critics: CriticPair,
    classifier: Classifier,
    actor: ContinuousActor | DiscreteActor,
    learning_state: OptState,
    learner: Learner,
    lagrangians: jax.Array,
    safety_discount: float,
    key: jax.Array,
) -> tuple[tuple[CriticPair, OptState], jax.Array]:
    terminals = jnp.zeros_like(batch.reward)

    def loss_fn(safety_critics):
        #sample action, based on observation o
        sample_log_prob = jax.vmap(lambda o: actor(o).sample_and_log_prob(seed=key))
        next_action, log_pi = sample_log_prob(batch.next_observation)
        #compute value of Q for the next action
        next_qs = jax.vmap(target_safety_critics)(batch.next_observation, next_action)
        debiased_q = jnp.minimum(*next_qs)
        #compute log term of policy probability
        '''safety_logit = jax.vmap(classifier)(batch.observation, batch.action)
        safety_logit = jnp.clip(safety_logit, -100, None) #clip safety_logit

        safety_prob = jax.nn.sigmoid(safety_logit)'''
        X_input_format = jnp.hstack((batch.observation, batch.action))
        #X_input_np = X_input_format.copy().block_until_ready()        
        safety_prob = classifier.predict_proba(X_input_format)
        log_safety_prob = jnp.log(safety_prob) #jnp.log(safety_prob) #classifier probability for safety #const: jnp.log(1.0)
        log_safety_prob = jnp.clip(log_safety_prob, -100, None) #clip probability
        next_q = log_safety_prob + safety_discount * debiased_q
        next_q = jax.lax.stop_gradient(next_q)
        #computes critic networks prediction. 
        q1, q2 = jax.vmap(safety_critics)(batch.observation, batch.action)
        #compute loss for both critic networks
        return (l2_loss(q1, next_q) + l2_loss(q2, next_q)).mean()
    #do optimization step for critic network (NOT target critic)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(safety_critics)
    new_critics, new_learning_state = learner.grad_step(safety_critics, grads, learning_state)
    return (new_critics, new_learning_state), loss
#-----------------------------------

@eqx.filter_jit
def update_actor(
    batch: Transition,
    actor: ContinuousActor | DiscreteActor,
    critics: CriticPair,
    safety_critics: CriticPair,
    learning_state: OptState,
    learner: Learner,
    lagrangians: jax.Array,
    key: jax.Array,
) -> tuple[tuple[ContinuousActor | DiscreteActor, OptState], dict[str, jax.Array]]:
    def loss_fn(actor):
        sample_log_prob = jax.vmap(lambda o: actor(o).sample_and_log_prob(seed=key))
        action, log_pi = sample_log_prob(batch.observation)
        qs = jax.vmap(critics)(batch.observation, action)
        debiased_q = jnp.minimum(*qs)
        safety_qs = jax.vmap(safety_critics)(batch.observation, action)
        debiased_safety_q = jnp.minimum(*safety_qs)
        surprise = -lagrangians * log_pi
        #add term for safety critic to objective (subtract safety critic q value)
        objective = debiased_q + debiased_safety_q + surprise #added safety critic term 
        return -objective.mean(), (log_pi.mean(), jnp.mean(debiased_q), jnp.mean(debiased_safety_q), jnp.mean(surprise))

    (loss, (log_pi, critic_component, safety_critic_component, surprise_component)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(actor)
    new_actor, new_learning_state = learner.grad_step(actor, grads, learning_state)
    return (new_actor, new_learning_state), dict(loss=loss, log_pi=log_pi, critic_component=critic_component, safety_critic_component=safety_critic_component, surprise_component=surprise_component)


@eqx.filter_jit
def update_log_lagrangians(
    log_lagrangians: jax.Array,
    log_pi: jax.Array,
    target: jax.Array,
    learning_state: OptState,
    learner: Learner,
):
    def loss_fn(log_lagrangians):
        lagrangians = jnp.exp(log_lagrangians)
        penalty = -lagrangians * (log_pi - target)
        return penalty.sum()

    loss, grads = eqx.filter_value_and_grad(loss_fn)(log_lagrangians)
    new_log_lagrangians, new_learning_state = learner.grad_step(
        log_lagrangians, grads, learning_state
    )
    return (new_log_lagrangians, new_learning_state), loss
