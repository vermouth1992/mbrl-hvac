import numpy as np
import torch
from torchlib.common import convert_numpy_to_tensor, FloatTensor
from torchlib.deep_rl.algorithm.model_based import BestRandomActionPlanner


class BestRandomActionHistoryPlanner(BestRandomActionPlanner):
    """
    The only difference is that the input state and action contains T time steps
    """

    def predict(self, history_state, history_actions, current_state):
        """

        Args:
            history_state: (T - 1, 6)
            history_actions: (T - 1, 4)
            current_state: (6,)

        Returns: best action (4,)

        """
        states = np.expand_dims(history_state, axis=0)  # (1, T - 1, 6)
        states = np.tile(states, (self.num_random_action_selection, 1, 1))  # (N, T - 1, 6)
        states = convert_numpy_to_tensor(states)

        next_states = np.expand_dims(current_state, axis=0)  # (1, 6)
        next_states = np.tile(next_states, (self.num_random_action_selection, 1))  # (N, 6)
        next_states = convert_numpy_to_tensor(next_states)

        actions = self.action_sampler.sample((self.horizon, self.num_random_action_selection))  # (H, N, 4)
        actions = convert_numpy_to_tensor(actions)

        history_actions = np.expand_dims(history_actions, axis=0)  # (1, T - 1, 4)
        current_action = np.tile(history_actions, (self.num_random_action_selection, 1, 1))  # (N, T - 1, 4)
        current_action = convert_numpy_to_tensor(current_action)

        with torch.no_grad():
            cost = torch.zeros(size=(self.num_random_action_selection,)).type(FloatTensor)
            for i in range(self.horizon):
                states = torch.cat((states, torch.unsqueeze(next_states, dim=1)), dim=1)  # # (N, T, 6)
                current_action = torch.cat((current_action, torch.unsqueeze(actions[i], dim=1)), dim=1)  # (N, T, 4)
                next_states = self.model.predict_next_states(states, current_action)  # (N, 6)
                cost += self.cost_fn(states[:, -1, :], actions[i], next_states) * self.gamma_inverse
                current_action = current_action[:, 1:, :]  # (N, T - 1, 4)
                states = states[:, 1:, :]

            best_action = actions[0, torch.argmin(cost, dim=0)]
            best_action = best_action.cpu().numpy()
            return best_action
