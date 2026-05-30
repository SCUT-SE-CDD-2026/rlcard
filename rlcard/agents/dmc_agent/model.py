# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
# Licensed under the Apache License, Version 2.0

import numpy as np

import torch
from torch import nn


class DMCNet(nn.Module):
    def __init__(self, state_shape, action_shape, mlp_layers=[512, 512, 512, 512, 512]):
        super().__init__()
        input_dim = np.prod(state_shape) + np.prod(action_shape)
        layer_dims = [input_dim] + mlp_layers
        fc = []
        for i in range(len(layer_dims) - 1):
            fc.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))
            fc.append(nn.ReLU())
        fc.append(nn.Linear(layer_dims[-1], 1))
        self.fc_layers = nn.Sequential(*fc)

    def forward(self, obs, actions):
        obs = torch.flatten(obs, 1)
        actions = torch.flatten(actions, 1)
        x = torch.cat((obs, actions), dim=1)
        values = self.fc_layers(x).flatten()
        return values


class DMCNetV2(nn.Module):
    """ChuDaDi V2 Q-network: separate state/action towers plus shared GRU history encoder."""

    def __init__(
        self,
        state_shape,
        action_shape,
        history_shape=(3, 13, 52),
        state_layers=(512, 512, 512),
        action_layers=(512, 512),
        gru_hidden_size=256,
        fusion_layers=(1024, 512, 256),
    ):
        super().__init__()
        self.history_shape = tuple(history_shape)
        history_players, _, history_dim = self.history_shape
        self.history_players = history_players
        self.state_tower = self._make_mlp(int(np.prod(state_shape)), state_layers)
        self.action_tower = self._make_mlp(int(np.prod(action_shape)), action_layers)
        self.history_gru = nn.GRU(
            input_size=history_dim,
            hidden_size=gru_hidden_size,
            batch_first=True,
        )
        fusion_input_dim = state_layers[-1] + action_layers[-1] + history_players * gru_hidden_size
        self.fusion_head = self._make_mlp(fusion_input_dim, fusion_layers, output_dim=1)

    @staticmethod
    def _make_mlp(input_dim, hidden_layers, output_dim=None):
        dims = [input_dim] + list(hidden_layers)
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
        if output_dim is not None:
            layers.append(nn.Linear(dims[-1], output_dim))
        return nn.Sequential(*layers)

    def forward(self, obs, actions, history):
        obs = torch.flatten(obs, 1)
        actions = torch.flatten(actions, 1)
        batch_size = history.shape[0]
        history = history.float().reshape(
            batch_size * self.history_players,
            self.history_shape[1],
            self.history_shape[2],
        )
        _, hidden = self.history_gru(history)
        history_latent = hidden[-1].reshape(batch_size, self.history_players * hidden.shape[-1])
        x = torch.cat(
            (
                self.state_tower(obs),
                self.action_tower(actions),
                history_latent,
            ),
            dim=1,
        )
        return self.fusion_head(x).flatten()


class DMCAgent:
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        exp_epsilon=0.01,
        device="0",
        model_version="v1",
        history_shape=None,
    ):
        self.use_raw = False
        self.device = "cuda:" + device if device != "cpu" else "cpu"
        self.model_version = model_version
        self.history_shape = history_shape
        if model_version == "v2":
            if history_shape is None:
                raise ValueError("history_shape is required for DMC model_version='v2'")
            self.net = DMCNetV2(state_shape, action_shape, history_shape).to(self.device)
        else:
            self.net = DMCNet(state_shape, action_shape, mlp_layers).to(self.device)
        self.exp_epsilon = exp_epsilon
        self.action_shape = action_shape

    def step(self, state):
        action_keys, values = self.predict(state)
        if self.exp_epsilon > 0 and np.random.rand() < self.exp_epsilon:
            action = np.random.choice(action_keys)
        else:
            action_idx = np.argmax(values)
            action = action_keys[action_idx]
        return action

    def eval_step(self, state):
        action_keys, values = self.predict(state)
        action_idx = np.argmax(values)
        action = action_keys[action_idx]
        info = {
            "values": {
                state["raw_legal_actions"][i]: float(values[i])
                for i in range(len(action_keys))
            }
        }
        return action, info

    def share_memory(self):
        self.net.share_memory()

    def eval(self):
        self.net.eval()

    def parameters(self):
        return self.net.parameters()

    def predict(self, state):
        obs = state["obs"].astype(np.float32)
        legal_actions = state["legal_actions"]
        action_keys = np.array(list(legal_actions.keys()))
        action_values = list(legal_actions.values())
        for i in range(len(action_values)):
            if action_values[i] is None:
                action_values[i] = np.zeros(self.action_shape[0])
                action_values[i][action_keys[i]] = 1
        action_values = np.array(action_values, dtype=np.float32)
        obs = np.repeat(obs[np.newaxis, :], len(action_keys), axis=0)
        with torch.inference_mode():
            if self.model_version == "v2":
                history = np.repeat(
                    state["history"].astype(np.float32)[np.newaxis, :],
                    len(action_keys),
                    axis=0,
                )
                values = self.net.forward(
                    torch.from_numpy(obs).to(self.device),
                    torch.from_numpy(action_values).to(self.device),
                    torch.from_numpy(history).to(self.device),
                )
            else:
                values = self.net.forward(
                    torch.from_numpy(obs).to(self.device),
                    torch.from_numpy(action_values).to(self.device),
                )
        return action_keys, values.cpu().detach().numpy()

    def forward(self, obs, actions, history=None):
        if self.model_version == "v2":
            if history is None:
                raise ValueError("history is required for DMC V2 forward")
            return self.net.forward(obs, actions, history)
        return self.net.forward(obs, actions)

    def load_state_dict(self, state_dict):
        return self.net.load_state_dict(state_dict)

    def state_dict(self):
        return self.net.state_dict()

    def set_device(self, device):
        self.device = "cuda:" + str(device) if str(device) != "cpu" else "cpu"
        self.net.to(self.device)


class DMCAgentV2(DMCAgent):
    def __init__(self, state_shape, action_shape, history_shape, exp_epsilon=0.01, device="0"):
        super().__init__(
            state_shape=state_shape,
            action_shape=action_shape,
            exp_epsilon=exp_epsilon,
            device=device,
            model_version="v2",
            history_shape=history_shape,
        )


class DMCModel:
    def __init__(
        self,
        state_shape,
        action_shape,
        mlp_layers=[512, 512, 512, 512, 512],
        exp_epsilon=0.01,
        device=0,
        model_version="v1",
        history_shape=None,
    ):
        self.model_version = model_version
        self.agents = []
        for player_id in range(len(state_shape)):
            agent = DMCAgent(
                state_shape[player_id],
                action_shape[player_id],
                mlp_layers,
                exp_epsilon,
                str(device),
                model_version=model_version,
                history_shape=history_shape[player_id] if history_shape else None,
            )
            self.agents.append(agent)

    def share_memory(self):
        for agent in self.agents:
            agent.share_memory()

    def eval(self):
        for agent in self.agents:
            agent.eval()

    def parameters(self, index):
        return self.agents[index].parameters()

    def get_agent(self, index):
        return self.agents[index]

    def get_agents(self):
        return self.agents
