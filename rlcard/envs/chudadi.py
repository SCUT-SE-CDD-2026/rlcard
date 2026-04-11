from collections import OrderedDict
import numpy as np

from rlcard.envs import Env
from rlcard.games.chudadi.utils import (
    action_id_to_cards,
    action_to_feature_meta,
    card_to_id,
    cards_to_str,
)


class ChudadiEnv(Env):
    def __init__(self, config):
        from rlcard.games.chudadi import Game

        self.name = "chudadi"
        # Northern rule is the default
        self.northern_rule = config.get("northern_rule", True)
        self.game = Game(northern_rule=self.northern_rule)
        super().__init__(config)

        self._action_types = [
            "none",
            "single",
            "pair",
            "triple",
            "straight",
            "flush",
            "full_house",
            "four_of_a_kind",  # 4+1 铁支
            "straight_flush",
            "bomb",  # 4张裸出 炸弹
        ]
        self._action_type_to_index = {
            name: idx for idx, name in enumerate(self._action_types)
        }
        self._bit_positions = np.arange(52, dtype=np.uint64)

        # 332 + 2 (triple + bomb type) + 1 (is_northern_rule flag) = 335
        self.state_shape = [[335] for _ in range(self.num_players)]
        self.action_shape = [[140] for _ in range(self.num_players)]

    def _extract_state(self, state):
        player_id = state["current_player"]

        current_hand = self._cards_to_array(state["current_hand"])
        last_action = self._cards_to_array(state["last_action"])

        action_type = state["last_action_type"] or "none"
        action_type_one_hot = np.zeros(len(self._action_types), dtype=np.int8)
        action_type_one_hot[self._action_type_to_index[action_type]] = 1

        action_length_one_hot = np.zeros(14, dtype=np.int8)
        action_length = state["last_action_length"]
        if 0 <= action_length < 14:
            action_length_one_hot[action_length] = 1

        is_leader = 1 if action_length == 0 else 0
        leader_relative_pos = np.zeros(3, dtype=np.int8)
        leader_id = state["last_player"]
        if not is_leader and leader_id is not None and leader_id != player_id:
            for rel_index, offset in enumerate((1, 2, 3)):
                if leader_id == self._relative_player_id(player_id, offset):
                    leader_relative_pos[rel_index] = 1
                    break

        next_cards_left = self._get_relative_cards_left_one_hot(state, player_id, 1)
        across_cards_left = self._get_relative_cards_left_one_hot(state, player_id, 2)
        prev_cards_left = self._get_relative_cards_left_one_hot(state, player_id, 3)

        history_next = self._get_relative_played_history(state, player_id, 1)
        history_across = self._get_relative_played_history(state, player_id, 2)
        history_prev = self._get_relative_played_history(state, player_id, 3)

        relative_pass_mask = self._get_relative_pass_mask(state, player_id)
        is_next_warning = (
            1
            if state["num_cards_left"][self._relative_player_id(player_id, 1)] == 1
            else 0
        )

        is_northern_rule = 1 if self.northern_rule else 0

        obs = np.concatenate(
            [
                current_hand,
                last_action,
                action_type_one_hot,
                action_length_one_hot,
                leader_relative_pos,
                next_cards_left,
                across_cards_left,
                prev_cards_left,
                history_next,
                history_across,
                history_prev,
                np.asarray([is_leader], dtype=np.int8),
                relative_pass_mask,
                np.asarray([is_next_warning], dtype=np.int8),
                np.asarray([is_northern_rule], dtype=np.int8),
            ]
        )

        extracted_state = OrderedDict(
            {
                "obs": obs,
                "legal_actions": self._get_legal_actions(
                    current_hand=state["current_hand"],
                    action_ids=state["actions"],
                ),
            }
        )
        extracted_state["raw_obs"] = state
        extracted_state["raw_legal_actions"] = list(state["raw_legal_actions"])
        extracted_state["action_record"] = self.action_recorder
        return extracted_state

    def get_payoffs(self):
        return self.game.get_payoffs()

    def _decode_action(self, action_id):
        if action_id == 0:
            return "pass"
        return action_id_to_cards(action_id)

    def _get_legal_actions(self, current_hand=None, action_ids=None):
        legal_actions = action_ids
        if legal_actions is None:
            legal_actions = self.game.state["actions"]
        if not legal_actions:
            return {}
        if current_hand is None:
            current_hand = self.game.state.get("current_hand", [])
        action_ids = np.asarray(legal_actions, dtype=np.uint64)
        features = self._action_ids_to_features(action_ids, current_hand)
        return {
            int(action_id): features[index]
            for index, action_id in enumerate(action_ids)
        }

    def get_perfect_information(self):
        state = {}
        state["hand_cards"] = [
            cards_to_str(player.current_hand) for player in self.game.players
        ]
        state["trace"] = list(self.game.state["trace"])
        state["current_player"] = self.game.round.current_player
        state["legal_actions"] = list(self.game.state["raw_legal_actions"])
        return state

    def get_action_feature(self, action, state=None):
        if state is None:
            current_hand = self.game.state.get("current_hand", [])
        else:
            raw_state = state.get("raw_obs", state)
            current_hand = raw_state.get("current_hand", [])
        features = self._action_ids_to_features(
            np.asarray([action], dtype=np.uint64),
            current_hand,
        )
        if features.size == 0:
            return np.zeros(self.action_shape[0], dtype=np.int8)
        return features[0]

    def _cards_to_array(self, cards):
        array = np.zeros(52, dtype=np.int8)
        for card in cards:
            array[card_to_id(card)] = 1
        return array

    def _relative_player_id(self, player_id, offset):
        return (player_id + offset) % self.num_players

    def _get_relative_cards_left_one_hot(self, state, player_id, offset):
        relative_id = self._relative_player_id(player_id, offset)
        count = state["num_cards_left"][relative_id]
        one_hot = np.zeros(14, dtype=np.int8)
        if 0 <= count < 14:
            one_hot[count] = 1
        return one_hot

    def _get_relative_played_history(self, state, player_id, offset):
        relative_id = self._relative_player_id(player_id, offset)
        return self._cards_to_array(state["played_cards"][relative_id])

    def _get_relative_pass_mask(self, state, player_id):
        mask = np.zeros(3, dtype=np.int8)
        if state["last_action_length"] == 0:
            return mask
        trace = state.get("trace") or []
        if not trace:
            return mask
        last_non_pass_idx = None
        for idx in range(len(trace) - 1, -1, -1):
            if trace[idx][1] != "pass":
                last_non_pass_idx = idx
                break
        if last_non_pass_idx is None:
            return mask
        passed_players = {
            pid for pid, action in trace[last_non_pass_idx + 1 :] if action == "pass"
        }
        for rel_index, offset in enumerate((1, 2, 3)):
            if self._relative_player_id(player_id, offset) in passed_players:
                mask[rel_index] = 1
        return mask

    def _action_ids_to_features(self, action_ids, current_hand):
        if action_ids.size == 0:
            return np.zeros((0, self.action_shape[0][0]), dtype=np.int8)
        action_bits = self._action_ids_to_arrays(action_ids)
        current_hand_bits = (
            self._cards_to_array(current_hand)
            if current_hand is not None
            else np.zeros(52, dtype=np.int8)
        )
        next_hand_bits = np.clip(
            current_hand_bits[np.newaxis, :] - action_bits, 0, 1
        ).astype(np.int8)
        action_type_features = np.zeros(
            (len(action_ids), len(self._action_types)), dtype=np.int8
        )
        action_main_rank = np.zeros((len(action_ids), 13), dtype=np.int8)
        action_kicker_rank = np.zeros((len(action_ids), 13), dtype=np.int8)
        for index, action_id in enumerate(action_ids):
            action_type = "none"
            main_idx = None
            kicker_idx = None
            if int(action_id) != 0:
                cards = action_id_to_cards(int(action_id))
                action_type, main_idx, kicker_idx = action_to_feature_meta(cards)
            action_type_features[index, self._action_type_to_index[action_type]] = 1
            if main_idx is not None:
                action_main_rank[index, main_idx] = 1
            if kicker_idx is not None:
                action_kicker_rank[index, kicker_idx] = 1
        return np.concatenate(
            [
                action_bits,
                next_hand_bits,
                action_type_features,
                action_main_rank,
                action_kicker_rank,
            ],
            axis=1,
        )

    def _action_ids_to_arrays(self, action_ids):
        if action_ids.size == 0:
            return np.zeros((0, 52), dtype=np.int8)
        bits = ((action_ids[:, None] >> self._bit_positions) & 1).astype(np.int8)
        return bits
