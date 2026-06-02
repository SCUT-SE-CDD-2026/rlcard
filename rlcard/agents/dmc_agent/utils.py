# Copyright 2021 RLCard Team of Texas A&M University
# Copyright 2021 DouZero Team of Kwai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
try:
    import resource
except ImportError:  # Windows does not provide `resource`.
    resource = None
import traceback

import numpy as np
import torch

shandle = logging.StreamHandler()
shandle.setFormatter(
    logging.Formatter(
        '[%(levelname)s:%(process)d %(module)s:%(lineno)d %(asctime)s] '
        '%(message)s'))
log = logging.getLogger('doudzero')
log.propagate = False
log.addHandler(shandle)
log.setLevel(logging.INFO)

def get_batch(
    free_queue,
    full_queue,
    buffers,
    batch_size,
    lock
):
    with lock:
        indices = [full_queue.get() for _ in range(batch_size)]
    batch = {
        key: torch.stack([buffers[key][m] for m in indices], dim=1)
        for key in buffers
    }
    for m in indices:
        free_queue.put(m)
    return batch

def _cap_num_buffers(num_buffers, per_buffer_objects):
    if resource is None:
        return num_buffers
    try:
        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
    except Exception:
        return num_buffers
    reserve = 16
    if soft_limit <= reserve:
        return max(1, min(num_buffers, 1))
    max_buffers = max(1, (soft_limit - reserve) // per_buffer_objects)
    if max_buffers < num_buffers:
        log.warning(
            "Capping num_buffers from %d to %d due to RLIMIT_NOFILE=%d. "
            "Consider increasing ulimit -Sn (e.g., 4096 65535).",
            num_buffers,
            max_buffers,
            soft_limit,
        )
        return max_buffers
    return num_buffers


def create_buffers(
    T,
    num_buffers,
    state_shape,
    action_shape,
    device_iterator,
    pin_memory=False,
    history_shape=None,
    strategic_reward_metrics=False,
):
    devices = list(device_iterator)
    num_players = len(state_shape)
    spec_count = 6 if history_shape is not None else 5
    if strategic_reward_metrics:
        spec_count += 2
    per_buffer_objects = max(1, len(devices) * num_players * spec_count)
    num_buffers = _cap_num_buffers(num_buffers, per_buffer_objects)
    buffers = {}
    for device in devices:
        buffers[device] = []
        for player_id in range(len(state_shape)):
            specs = dict(
                done=dict(size=(T,), dtype=torch.bool),
                episode_return=dict(size=(T,), dtype=torch.float32),
                target=dict(size=(T,), dtype=torch.float32),
                state=dict(size=(T,) + tuple(state_shape[player_id]), dtype=torch.int8),
                action=dict(size=(T,) + tuple(action_shape[player_id]), dtype=torch.int8),
            )
            if strategic_reward_metrics:
                specs["strategic_bonus_ratio"] = dict(size=(T,), dtype=torch.float32)
                specs["strategic_win"] = dict(size=(T,), dtype=torch.float32)
            if history_shape is not None:
                specs["history"] = dict(
                    size=(T,) + tuple(history_shape[player_id]),
                    dtype=torch.int8,
                )
            _buffers = {key: [] for key in specs}
            use_pin_memory = pin_memory
            for _ in range(num_buffers):
                for key in _buffers:
                    try:
                        _buffer = torch.empty(
                            **specs[key],
                            pin_memory=use_pin_memory,
                        ).share_memory_()
                    except (TypeError, RuntimeError):
                        if use_pin_memory:
                            log.warning(
                                "Pinned memory is not available. Falling back to regular CPU buffers."
                            )
                            use_pin_memory = False
                            _buffer = torch.empty(**specs[key]).share_memory_()
                        else:
                            raise
                    _buffers[key].append(_buffer)
            buffers[device].append(_buffers)
    return buffers, num_buffers

def create_optimizers(
    num_players,
    learning_rate,
    momentum,
    epsilon,
    alpha,
    learner_model
):
    optimizers = []
    for player_id in range(num_players):
        optimizer = torch.optim.RMSprop(
            learner_model.parameters(player_id),
            lr=learning_rate,
            momentum=momentum,
            eps=epsilon,
            alpha=alpha)
        optimizers.append(optimizer)
    return optimizers

def act(
    i,
    device,
    T,
    free_queue,
    full_queue,
    model,
    buffers,
    env
):
    try:
        log.info('Device %s Actor %i started.', str(device), i)

        # Configure environment
        env.seed(i)
        env.set_agents(model.get_agents())

        done_buf = [[] for _ in range(env.num_players)]
        episode_return_buf = [[] for _ in range(env.num_players)]
        target_buf = [[] for _ in range(env.num_players)]
        state_buf = [[] for _ in range(env.num_players)]
        action_buf = [[] for _ in range(env.num_players)]
        history_buf = [[] for _ in range(env.num_players)]
        strategic_bonus_ratio_buf = [[] for _ in range(env.num_players)]
        strategic_win_buf = [[] for _ in range(env.num_players)]
        has_history = hasattr(env, "history_shape")
        has_strategic_reward_metrics = getattr(env, "reward_mode", None) == "strategic_win_zero_sum"
        size = [0 for _ in range(env.num_players)]

        while True:
            trajectories, payoffs = env.run(is_training=True)
            reward_info = env.get_reward_info() if has_strategic_reward_metrics else None
            for p in range(env.num_players):
                size[p] += len(trajectories[p][:-1]) // 2
                diff = size[p] - len(target_buf[p])
                if diff > 0:
                    done_buf[p].extend([False for _ in range(diff-1)])
                    done_buf[p].append(True)
                    episode_return_buf[p].extend([0.0 for _ in range(diff-1)])
                    episode_return_buf[p].append(float(payoffs[p]))
                    target_buf[p].extend([float(payoffs[p]) for _ in range(diff)])
                    if has_strategic_reward_metrics:
                        is_winner = reward_info["winner"] == p
                        bonus_ratio = reward_info["strategic_bonus_ratio"] if is_winner else 0.0
                        strategic_win_buf[p].extend([0.0 for _ in range(diff-1)])
                        strategic_win_buf[p].append(1.0 if is_winner else 0.0)
                        strategic_bonus_ratio_buf[p].extend([0.0 for _ in range(diff-1)])
                        strategic_bonus_ratio_buf[p].append(float(bonus_ratio))
                    # State and action
                    for i in range(0, len(trajectories[p])-2, 2):
                        state = trajectories[p][i]
                        obs = state['obs']
                        action = env.get_action_feature(trajectories[p][i+1], state)
                        state_buf[p].append(torch.from_numpy(obs))
                        action_buf[p].append(torch.from_numpy(action))
                        if has_history:
                            history_buf[p].append(torch.from_numpy(state['history']))
                
                while size[p] > T:
                    index = free_queue[p].get()
                    if index is None:
                        break
                    for t in range(T):
                        buffers[p]['done'][index][t, ...] = done_buf[p][t]
                        buffers[p]['episode_return'][index][t, ...] = episode_return_buf[p][t]
                        buffers[p]['target'][index][t, ...] = target_buf[p][t]
                        buffers[p]['state'][index][t, ...] = state_buf[p][t]
                        buffers[p]['action'][index][t, ...] = action_buf[p][t]
                        if has_history:
                            buffers[p]['history'][index][t, ...] = history_buf[p][t]
                        if has_strategic_reward_metrics:
                            buffers[p]['strategic_bonus_ratio'][index][t, ...] = strategic_bonus_ratio_buf[p][t]
                            buffers[p]['strategic_win'][index][t, ...] = strategic_win_buf[p][t]
                    full_queue[p].put(index)
                    done_buf[p] = done_buf[p][T:]
                    episode_return_buf[p] = episode_return_buf[p][T:]
                    target_buf[p] = target_buf[p][T:]
                    state_buf[p] = state_buf[p][T:]
                    action_buf[p] = action_buf[p][T:]
                    if has_history:
                        history_buf[p] = history_buf[p][T:]
                    if has_strategic_reward_metrics:
                        strategic_bonus_ratio_buf[p] = strategic_bonus_ratio_buf[p][T:]
                        strategic_win_buf[p] = strategic_win_buf[p][T:]
                    size[p] -= T

    except KeyboardInterrupt:
        pass
    except Exception as e:
        log.error('Exception in worker process %i', i)
        traceback.print_exc()
        print()
        raise e
