"""
Planner for model-based RL
"""

import numpy as np
import torch
import torch.nn as nn

from torchlib.common import convert_numpy_to_tensor, FloatTensor
from torchlib.deep_rl import BaseAgent
from torchlib.utils.random.sampler import BaseSampler, IntSampler
from .world_model import WorldModel


class Planner(BaseAgent):
    """
    Planner predict the next best action given current state using model. Planner typically doesn't have memory.
    """

    def __init__(self, model: WorldModel):
        self.model = model

    def predict(self, state):
        raise NotImplementedError


class BestRandomActionPlanner(Planner):
    def __init__(self, model, action_sampler: BaseSampler, cost_fn=None,
                 horizon=15, num_random_action_selection=4096, gamma=0.95):
        """

        Args:
            model: Model instance. Can predict next states and optional cost (reward)
            action_sampler: Sampler that can sample actions
            cost_fn: if None, we expect model predicts both next states and cost (negative reward)
            horizon: Number of steps that we look into the future
            num_random_action_selection: Number of trajectories to sample
        """
        super(BestRandomActionPlanner, self).__init__(model=model)
        self.action_sampler = action_sampler
        self.horizon = horizon
        self.num_random_action_selection = num_random_action_selection
        self.gamma_inverse = 1. / gamma
        if cost_fn is None:
            self.cost_fn = model.cost_fn
        else:
            self.cost_fn = cost_fn

    def predict(self, state):
        states = np.expand_dims(state, axis=0)
        actions = self.action_sampler.sample((self.horizon, self.num_random_action_selection))
        states = np.tile(states, (self.num_random_action_selection, 1))
        states = convert_numpy_to_tensor(states)
        actions = convert_numpy_to_tensor(actions)

        with torch.no_grad():
            cost = torch.zeros(size=(self.num_random_action_selection,)).type(FloatTensor)
            for i in range(self.horizon):
                next_states = self.model.predict_next_states(states, actions[i])
                cost += self.cost_fn(states, actions[i], next_states) * self.gamma_inverse
                states = next_states

            best_action = actions[0, torch.argmin(cost, dim=0)]
            best_action = best_action.cpu().numpy()
            return best_action


class TanhActionModule(nn.Module):
    def __init__(self, init_action):
        super(TanhActionModule, self).__init__()
        init_action = convert_numpy_to_tensor(init_action)
        self.action = nn.Parameter(data=init_action, requires_grad=True)

    def forward(self, h):
        return torch.tanh(self.action[h: h + 1])


class GradientDescentActionPlanner(Planner):
    """
    Notes: only applicable to continuous action space. It also requires that cost_fn is differentiable.
    """

    def __init__(self, model, action_sampler, cost_fn=None, horizon=15, num_iterations=100, gamma=0.95):
        super(GradientDescentActionPlanner, self).__init__(model=model)
        self.action_sampler = action_sampler
        self.horizon = horizon
        self.num_iterations = num_iterations
        self.gamma_inverse = 1. / gamma
        if cost_fn is None:
            self.cost_fn = model.cost_fn
        else:
            self.cost_fn = cost_fn

    def predict(self, state):
        """ The model must be in evaluation mode and turn off gradient update

        Args:
            state: (ob_dim)

        Returns: optimal action (ac_dim)

        """
        action_module = TanhActionModule(init_action=self.action_sampler.sample((self.horizon,)))
        optimizer = torch.optim.Adam(action_module.parameters(), lr=1e-3)
        # t = tqdm(range(self.num_iterations), desc='Planning')
        t = range(self.num_iterations)
        for iteration in t:
            optimizer.zero_grad()
            cost = []
            current_state = convert_numpy_to_tensor(np.expand_dims(state, axis=0))
            for h in range(self.horizon):
                current_action = action_module.forward(h)
                next_states = self.model.predict_next_states(current_state, current_action)
                cost.append(self.cost_fn(current_state, current_action, next_states) * self.gamma_inverse)
                current_state = next_states
            cost = torch.mean(torch.cat(cost))
            cost.backward()

            nn.utils.clip_grad_norm_(action_module.parameters(), max_norm=1.0)

            optimizer.step()

            # t.set_description('Iter {}/{}, Cost {:.4f}'.format(iteration + 1, self.num_iterations, cost.item()))

        return action_module.forward(0)[0].cpu().detach().numpy()


class GameState(object):
    def __init__(self, state, model: WorldModel, cost_fn, action_sampler: IntSampler, horizon):
        self.state = state
        self.model = model
        self.cost_fn = cost_fn
        self.action_sampler = action_sampler
        self.horizon = horizon

    def play(self, action):
        next_state = self.model.predict_next_state(self.state, action)
        return GameState(next_state, self.model, self.cost_fn, self.action_sampler, self.horizon - 1)

    def simulate(self):
        """ Simulate from current node. The default agent is random agent.

        Returns: a dictionary map from action to prior and value_estimate

        """
        # end of the game
        if self.horizon == 0:
            return [], 0.0

        child_priors = [1.0 for _ in range(self.action_sampler.high)]
        state = self.state
        states = []
        actions = []
        next_states = []
        for _ in range(self.horizon):
            action = self.action_sampler.sample(shape=None)
            states.append(state)
            actions.append(action)
            state = self.model.predict_next_state(state, action)
            next_states.append(state)

        states = np.array(states)
        actions = np.array(actions)
        next_states = np.array(next_states)
        cost = self.cost_fn(states, actions, next_states)
        cost = np.sum(cost)
        return child_priors, -cost


class UCTNode():
    def __init__(self, game_state: GameState, parent=None, prior=2.0):
        self.game_state = game_state
        self.is_expanded = False
        self.parent = parent  # Optional[UCTNode]
        self.children = {}  # Dict[move, UCTNode]
        self.prior = prior  # float
        self.total_value = 0  # float
        self.number_visits = 0  # int

    def Q(self):  # returns float
        return self.total_value / (1 + self.number_visits)

    def U(self):  # returns float
        return np.sqrt(np.log(self.parent.number_visits + 1) * self.prior / (1 + self.number_visits))

    def best_child(self):
        # no children
        if len(self.children) == 0:
            return None
        return max(self.children.values(),
                   key=lambda node: node.Q() + node.U())

    def select_leaf(self):
        current = self
        while current.is_expanded:
            best_child = current.best_child()
            if best_child is None:
                break
            else:
                current = best_child
        return current

    def expand(self, child_priors):
        self.is_expanded = True
        for move, prior in enumerate(child_priors):
            self.add_child(move, prior)

    def add_child(self, move, prior):
        self.children[move] = UCTNode(
            self.game_state.play(move), parent=self, prior=prior)

    def backup(self, value_estimate):
        current = self
        while current.parent is not None:
            current.number_visits += 1
            current.total_value += value_estimate
            current = current.parent


def UCT_search(game_state: GameState, num_reads):
    root = UCTNode(game_state)
    for _ in range(num_reads):
        leaf = root.select_leaf()
        child_priors, value_estimate = leaf.game_state.simulate()
        leaf.expand(child_priors)
        leaf.backup(value_estimate)
    return max(root.children.items(),
               key=lambda item: item[1].number_visits)


class UCTPlanner(Planner):
    def __init__(self, model, action_sampler: IntSampler, cost_fn=None, horizon=15, num_reads=1000):
        """ Upper bounded confidence Monte Carlo Tree Search (MCTS)

        Args:
            model: Model instance. Can predict next states and optional cost (reward)
            cost_fn: if None, we expect model predicts both next states and cost (negative reward)
            horizon: The deepest steps to simulate the actions
            num_reads: Number of times to run simulation to expand a node
        """
        super(UCTPlanner, self).__init__(model=model)
        assert isinstance(action_sampler, IntSampler), 'Action sampler must be IntSampler for UCT Planner'
        self.action_sampler = action_sampler
        if cost_fn is None:
            self.cost_fn = model.cost_fn
        else:
            self.cost_fn = cost_fn

        self.horizon = horizon
        self.num_reads = num_reads

    def predict(self, state):
        initial_state = GameState(state, self.model, self.cost_fn, self.action_sampler, horizon=self.horizon)
        action = UCT_search(initial_state, self.num_reads)[0]
        return action
