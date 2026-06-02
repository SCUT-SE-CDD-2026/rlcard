import unittest

import rlcard
from rlcard.games.base import Card
from rlcard.games.chudadi.judger import ChuDaDiJudger
from rlcard.games.chudadi.round import ChuDaDiRound
from rlcard.games.chudadi.player import ChuDaDiPlayer
from rlcard.games.chudadi.utils import (
    PASS_ACTION,
    RANK_TO_VALUE,
    START_CARD,
    can_beat,
    card_to_id,
    cards_to_action_id,
    get_legal_actions,
    make_action,
)


class TestChudadiEnv(unittest.TestCase):
    def setUp(self):
        self.env = rlcard.make("chudadi")

    @staticmethod
    def _make_cards(ranks):
        suits = ["D", "C", "H", "S"]
        return [Card(suits[i % len(suits)], rank) for i, rank in enumerate(ranks)]

    def _get_feature(self, action_cards, extra_cards=None, hand_override=None):
        hand = hand_override if hand_override is not None else list(action_cards) + (extra_cards or [])
        state = {"raw_obs": {"current_hand": hand}}
        action_id = cards_to_action_id(action_cards) if action_cards else 0
        feature = self.env.get_action_feature(action_id, state)
        return feature, hand

    def _assert_one_hot(self, segment, expected_index=None):
        if expected_index is None:
            self.assertEqual(int(segment.sum()), 0)
            return
        self.assertEqual(int(segment.sum()), 1)
        self.assertEqual(int(segment[expected_index]), 1)

    def _assert_action_future_bits(self, feature, action_cards, hand):
        action_set = {(card.suit, card.rank) for card in action_cards}
        action_bits = feature[:52]
        future_bits = feature[52:104]
        self.assertEqual(int(action_bits.sum()), len(action_cards))
        for card in action_cards:
            idx = card_to_id(card)
            self.assertEqual(int(action_bits[idx]), 1)
            self.assertEqual(int(future_bits[idx]), 0)
        extra = next((card for card in hand if (card.suit, card.rank) not in action_set), None)
        if extra is not None:
            idx = card_to_id(extra)
            self.assertEqual(int(action_bits[idx]), 0)
            self.assertEqual(int(future_bits[idx]), 1)

    def test_action_feature_meta(self):
        extra = [Card("C", "2")]
        cases = [
            ("single", [Card("S", "A")], 1, RANK_TO_VALUE["A"], None),
            ("pair", [Card("D", "5"), Card("S", "5")], 2, RANK_TO_VALUE["5"], None),
            ("triple", [Card("D", "8"), Card("H", "8"), Card("S", "8")], 3, RANK_TO_VALUE["8"], None),
            (
                "straight",
                [Card("D", "7"), Card("C", "8"), Card("H", "9"), Card("S", "T"), Card("D", "J")],
                4,
                RANK_TO_VALUE["J"],
                RANK_TO_VALUE["T"],
            ),
            (
                "flush",
                [Card("H", "3"), Card("H", "5"), Card("H", "7"), Card("H", "9"), Card("H", "J")],
                5,
                RANK_TO_VALUE["J"],
                RANK_TO_VALUE["9"],
            ),
            (
                "full_house",
                [Card("D", "K"), Card("C", "K"), Card("H", "K"), Card("S", "4"), Card("H", "4")],
                6,
                RANK_TO_VALUE["K"],
                RANK_TO_VALUE["4"],
            ),
            (
                "four_of_a_kind",
                [Card("D", "Q"), Card("C", "Q"), Card("H", "Q"), Card("S", "Q"), Card("D", "3")],
                7,
                RANK_TO_VALUE["Q"],
                RANK_TO_VALUE["3"],
            ),
            (
                "straight_flush",
                [Card("S", "9"), Card("S", "T"), Card("S", "J"), Card("S", "Q"), Card("S", "K")],
                8,
                RANK_TO_VALUE["K"],
                RANK_TO_VALUE["Q"],
            ),
        ]

        for _, cards, type_index, main_index, kicker_index in cases:
            feature, hand = self._get_feature(cards, extra_cards=extra)
            self.assertEqual(feature.size, 139)
            self._assert_action_future_bits(feature, cards, hand)
            self._assert_one_hot(feature[104:113], type_index)
            self._assert_one_hot(feature[113:126], main_index)
            self._assert_one_hot(feature[126:139], kicker_index)

    def test_action_feature_pass(self):
        feature, hand = self._get_feature([], extra_cards=[Card("D", "3")])
        self.assertEqual(feature.size, 139)
        self.assertEqual(int(feature[:52].sum()), 0)
        self._assert_one_hot(feature[104:113], 0)
        self._assert_one_hot(feature[113:126], None)
        self._assert_one_hot(feature[126:139], None)
        idx = card_to_id(hand[0])
        self.assertEqual(int(feature[52 + idx]), 1)

    def test_state_and_history_shapes(self):
        state, _ = self.env.reset()
        self.assertEqual(state["obs"].shape, (178,))
        self.assertEqual(state["history"].shape, (3, 13, 52))
        self.assertEqual(self.env.state_shape, [[178]] * 4)
        self.assertEqual(self.env.action_shape, [[139]] * 4)
        self.assertEqual(self.env.history_shape, [[3, 13, 52]] * 4)

    def test_v3_state_shape_and_cumulative_history(self):
        env = rlcard.make("chudadi", config={"obs_version": "v3"})
        state, _ = env.reset()
        self.assertEqual(state["obs"].shape, (334,))
        self.assertNotIn("history", state)
        self.assertEqual(env.state_shape, [[334]] * 4)
        self.assertEqual(env.action_shape, [[139]] * 4)
        self.assertFalse(hasattr(env, "history_shape"))

        raw_state = dict(state["raw_obs"])
        raw_state["current_player"] = 0
        raw_state["played_cards"] = [
            [Card("D", "3")],
            [Card("D", "4"), Card("S", "4")],
            [Card("H", "5")],
            [Card("C", "6")],
        ]
        extracted = env._extract_state(raw_state)
        history = extracted["obs"][172:328]
        self.assertEqual(int(history[:52].sum()), 2)
        self.assertEqual(int(history[52:104].sum()), 1)
        self.assertEqual(int(history[104:].sum()), 1)
        self.assertEqual(int(history[card_to_id(Card("D", "4"))]), 1)
        self.assertEqual(int(history[card_to_id(Card("S", "4"))]), 1)
        self.assertEqual(int(history[52 + card_to_id(Card("H", "5"))]), 1)
        self.assertEqual(int(history[104 + card_to_id(Card("C", "6"))]), 1)

    def test_baopei_can_be_disabled_in_round(self):
        for enabled, expected in ((True, 0), (False, None)):
            players = [ChuDaDiPlayer(pid) for pid in range(4)]
            players[0].set_current_hand([Card("D", "4"), Card("S", "2")])
            players[1].set_current_hand([Card("S", "5")])
            players[2].set_current_hand([Card("D", "6")])
            players[3].set_current_hand([Card("D", "7")])
            round_ = ChuDaDiRound(None, num_players=4, northern_rule=True, enable_baopei=enabled)
            round_.last_action = make_action([Card("D", "3")])
            round_.proceed_round(players[0], [Card("D", "4")], players)
            self.assertEqual(round_.baopei_player, expected)

    def test_v3_env_defaults_keep_baopei_enabled(self):
        env = rlcard.make("chudadi", config={"obs_version": "v3"})
        env.reset()
        self.assertTrue(env.game.round.enable_baopei)

        env = rlcard.make("chudadi", config={"obs_version": "v3", "enable_baopei": False})
        env.reset()
        self.assertFalse(env.game.round.enable_baopei)

    def test_first_trick_must_contain_d3(self):
        hand = [START_CARD, Card("S", "A"), Card("H", "K"), Card("C", "Q"), Card("D", "4"), Card("S", "5")]
        actions = get_legal_actions(hand, last_action=None, must_contain_card=True)
        self.assertTrue(actions)
        for action in actions:
            self.assertIn(START_CARD, action.cards)

        actions_no_rule = get_legal_actions(hand, last_action=None, must_contain_card=False)
        self.assertTrue(any(START_CARD not in action.cards for action in actions_no_rule))

    def test_no_naked_four_card_bomb_and_four_with_one_exists(self):
        hand = [
            Card("D", "7"), Card("C", "7"), Card("H", "7"), Card("S", "7"), Card("D", "3"), Card("C", "4"),
        ]
        actions = get_legal_actions(hand, last_action=None, must_contain_card=False)
        self.assertFalse(any(len(action.cards) == 4 for action in actions))
        self.assertTrue(any(action.action_type == "four_of_a_kind" and len(action.cards) == 5 for action in actions))

    def test_northern_cross_type_five_card_and_must_play(self):
        last_action = make_action([Card("D", "4"), Card("C", "5"), Card("H", "6"), Card("S", "7"), Card("D", "8")])
        hand = [Card("H", "3"), Card("H", "5"), Card("H", "7"), Card("H", "9"), Card("H", "J")]
        actions = get_legal_actions(hand, last_action, must_contain_card=False, northern_rule=True)
        self.assertTrue(actions)
        self.assertTrue(all(action.action_type != "pass" for action in actions))
        self.assertTrue(all(action.action_type == "flush" for action in actions))

    def test_southern_bomb_only_cross_type_behavior(self):
        full_house = make_action([Card("D", "4"), Card("C", "4"), Card("H", "4"), Card("S", "5"), Card("D", "5")])
        four_with_one = make_action([Card("D", "6"), Card("C", "6"), Card("H", "6"), Card("S", "6"), Card("D", "3")])
        straight_flush = make_action([Card("S", "7"), Card("S", "8"), Card("S", "9"), Card("S", "T"), Card("S", "J")])
        flush = make_action([Card("H", "3"), Card("H", "5"), Card("H", "7"), Card("H", "9"), Card("H", "J")])
        self.assertTrue(can_beat(four_with_one, full_house, northern_rule=False))
        self.assertTrue(can_beat(straight_flush, four_with_one, northern_rule=False))
        self.assertFalse(can_beat(flush, full_house, northern_rule=False))

    def test_must_play_if_can_beat(self):
        last_action = make_action([Card("D", "4"), Card("S", "4")])
        hand = [Card("D", "5"), Card("S", "5"), Card("H", "7")]
        actions = get_legal_actions(hand, last_action, must_contain_card=False)
        self.assertTrue(actions)
        self.assertTrue(all(action.action_type != "pass" for action in actions))

    def test_scoring_northern_rule(self):
        players = [ChuDaDiPlayer(pid) for pid in range(4)]
        players[0].set_current_hand([])
        players[1].set_current_hand(self._make_cards(["2", "2", "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4"]))
        players[2].set_current_hand(self._make_cards(["2", "A", "K", "Q", "J", "T", "9", "8", "7", "6"]))
        players[3].set_current_hand(self._make_cards(["A", "K", "Q"]))

        payoffs = ChuDaDiJudger(None, northern_rule=True).judge_payoffs(players, winner_id=0)
        self.assertEqual(payoffs[0], 75)
        self.assertEqual(payoffs[1], -52)
        self.assertEqual(payoffs[2], -20)
        self.assertEqual(payoffs[3], -3)

    def test_scoring_southern_holding_two_and_spade_two_winner_doubles(self):
        players = [ChuDaDiPlayer(pid) for pid in range(4)]
        players[0].set_current_hand([])
        players[1].set_current_hand([Card("D", "2"), Card("D", "3")])
        players[2].set_current_hand([Card("C", "4"), Card("D", "5")])
        players[3].set_current_hand([Card("C", "6")])
        winning_action = make_action([Card("S", "2")])
        payoffs = ChuDaDiJudger(None, northern_rule=False).judge_payoffs(
            players,
            winner_id=0,
            northern_rule=False,
            winning_action=winning_action,
        )
        self.assertEqual(payoffs, [14, -8, -4, -2])

    def test_baopei_moves_all_penalty_to_baopei_player(self):
        players = [ChuDaDiPlayer(pid) for pid in range(4)]
        players[0].set_current_hand([])
        players[1].set_current_hand([Card("D", "3")])
        players[2].set_current_hand([Card("D", "4"), Card("D", "5")])
        players[3].set_current_hand([Card("D", "6")])
        payoffs = ChuDaDiJudger(None, northern_rule=True).judge_payoffs(
            players,
            winner_id=0,
            northern_rule=True,
            baopei_player=2,
        )
        self.assertEqual(payoffs, [4, 0, -4, 0])

    def test_strategic_reward_three_round_suppression_and_multi_card_bonus(self):
        env = rlcard.make("chudadi", config={"reward_mode": "strategic_win_zero_sum"})
        env.reset()
        env.game.winner_id = 0
        env.game.round.trace = [
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D3 C3"),
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D4 C4"),
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D5"),
        ]

        payoffs = env.get_payoffs()

        self.assertAlmostEqual(payoffs[0], 1.4)
        for seat in (1, 2, 3):
            self.assertAlmostEqual(payoffs[seat], -1.4 / 3.0)

    def test_strategic_reward_advantage_when_others_play(self):
        env = rlcard.make("chudadi", config={"reward_mode": "strategic_win_zero_sum"})
        env.reset()
        env.game.winner_id = 0
        env.game.round.trace = [
            (1, "S3"),
            (2, "pass"),
            (3, "pass"),
            (0, "D3"),
            (1, "pass"),
            (2, "S4"),
            (3, "pass"),
            (0, "D4"),
            (1, "pass"),
            (2, "pass"),
            (3, "S5"),
            (0, "D5"),
        ]

        payoffs = env.get_payoffs()

        self.assertAlmostEqual(payoffs[0], 1.15)
        for seat in (1, 2, 3):
            self.assertAlmostEqual(payoffs[seat], -1.15 / 3.0)

    def test_strategic_reward_uses_highest_control_bonus_only(self):
        env = rlcard.make("chudadi", config={"reward_mode": "strategic_win_zero_sum"})
        env.reset()
        env.game.winner_id = 0
        env.game.round.trace = [
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D3"),
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D4"),
        ]

        payoffs = env.get_payoffs()

        self.assertAlmostEqual(payoffs[0], 1.2)
        for seat in (1, 2, 3):
            self.assertAlmostEqual(payoffs[seat], -1.2 / 3.0)

    def test_win_loss_reward_ignores_strategic_bonus(self):
        env = rlcard.make("chudadi", config={"reward_mode": "win_loss_zero_sum"})
        env.reset()
        env.game.winner_id = 0
        env.game.round.trace = [
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D3 C3"),
            (1, "pass"),
            (2, "pass"),
            (3, "pass"),
            (0, "D4"),
        ]

        self.assertEqual(env.get_payoffs(), [1.0, -1.0 / 3.0, -1.0 / 3.0, -1.0 / 3.0])

if __name__ == "__main__":
    unittest.main()
