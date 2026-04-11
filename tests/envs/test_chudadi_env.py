import unittest

import rlcard
from rlcard.games.base import Card
from rlcard.games.chudadi.judger import ChuDaDiJudger
from rlcard.games.chudadi.player import ChuDaDiPlayer
from rlcard.games.chudadi.utils import (
    PASS_ACTION,
    RANK_TO_VALUE,
    START_CARD,
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
        if hand_override is not None:
            hand = hand_override
        else:
            hand = list(action_cards) + (extra_cards or [])
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
        extra = None
        for card in hand:
            if (card.suit, card.rank) not in action_set:
                extra = card
                break
        if extra is not None:
            idx = card_to_id(extra)
            self.assertEqual(int(action_bits[idx]), 0)
            self.assertEqual(int(future_bits[idx]), 1)

    def test_action_feature_meta(self):
        extra = [Card("C", "2")]
        cases = [
            (
                "single",
                [Card("S", "A")],
                1,
                RANK_TO_VALUE["A"],
                None,
            ),
            (
                "pair",
                [Card("D", "5"), Card("S", "5")],
                2,
                RANK_TO_VALUE["5"],
                None,
            ),
            (
                "triple",
                [Card("D", "8"), Card("H", "8"), Card("S", "8")],
                3,
                RANK_TO_VALUE["8"],
                None,
            ),
            (
                "straight",
                [
                    Card("D", "7"),
                    Card("C", "8"),
                    Card("H", "9"),
                    Card("S", "T"),
                    Card("D", "J"),
                ],
                4,
                RANK_TO_VALUE["J"],
                RANK_TO_VALUE["T"],
            ),
            (
                "flush",
                [
                    Card("H", "3"),
                    Card("H", "5"),
                    Card("H", "7"),
                    Card("H", "9"),
                    Card("H", "J"),
                ],
                5,
                RANK_TO_VALUE["J"],
                RANK_TO_VALUE["9"],
            ),
            (
                "full_house",
                [
                    Card("D", "K"),
                    Card("C", "K"),
                    Card("H", "K"),
                    Card("S", "4"),
                    Card("H", "4"),
                ],
                6,
                RANK_TO_VALUE["K"],
                RANK_TO_VALUE["4"],
            ),
            (
                "four_of_a_kind",
                [
                    Card("D", "Q"),
                    Card("C", "Q"),
                    Card("H", "Q"),
                    Card("S", "Q"),
                    Card("D", "3"),
                ],
                7,
                RANK_TO_VALUE["Q"],
                RANK_TO_VALUE["3"],
            ),
            (
                "straight_flush",
                [
                    Card("S", "9"),
                    Card("S", "T"),
                    Card("S", "J"),
                    Card("S", "Q"),
                    Card("S", "K"),
                ],
                8,
                RANK_TO_VALUE["K"],
                RANK_TO_VALUE["Q"],
            ),
        ]

        for _, cards, type_index, main_index, kicker_index in cases:
            feature, hand = self._get_feature(cards, extra_cards=extra)
            self.assertEqual(feature.size, 140)
            self._assert_action_future_bits(feature, cards, hand)
            self._assert_one_hot(feature[104:114], type_index)
            self._assert_one_hot(feature[114:127], main_index)
            self._assert_one_hot(feature[127:140], kicker_index)

    def test_action_feature_pass(self):
        feature, hand = self._get_feature([], extra_cards=[Card("D", "3")])
        self.assertEqual(feature.size, 140)
        self.assertEqual(int(feature[:52].sum()), 0)
        self._assert_one_hot(feature[104:114], 0)
        self._assert_one_hot(feature[114:127], None)
        self._assert_one_hot(feature[127:140], None)
        idx = card_to_id(hand[0])
        self.assertEqual(int(feature[52 + idx]), 1)

    def test_first_trick_must_contain_d3(self):
        hand = [
            START_CARD,
            Card("S", "A"),
            Card("H", "K"),
            Card("C", "Q"),
            Card("D", "4"),
            Card("S", "5"),
        ]
        actions = get_legal_actions(hand, last_action=None, must_contain_card=True)
        self.assertTrue(actions)
        for action in actions:
            self.assertIn(START_CARD, action.cards)

        actions_no_rule = get_legal_actions(
            hand, last_action=None, must_contain_card=False
        )
        self.assertTrue(
            any(START_CARD not in action.cards for action in actions_no_rule)
        )

    def test_no_cross_type_beating(self):
        last_action = make_action([Card("D", "4"), Card("S", "4")])
        hand = [
            Card("D", "5"),
            Card("C", "6"),
            Card("H", "7"),
            Card("S", "8"),
            Card("D", "9"),
        ]
        actions = get_legal_actions(hand, last_action, must_contain_card=False)
        self.assertEqual(len(actions), 1)
        self.assertEqual(actions[0], PASS_ACTION)

    def test_must_play_if_can_beat(self):
        last_action = make_action([Card("D", "4"), Card("S", "4")])
        hand = [
            Card("D", "5"),
            Card("S", "5"),
            Card("H", "7"),
        ]
        actions = get_legal_actions(hand, last_action, must_contain_card=False)
        self.assertTrue(actions)
        self.assertTrue(all(action.action_type != "pass" for action in actions))

    def test_scoring_northern_rule(self):
        """Test northern rule scoring:
        - Winner: 0 points
        - >= 13 cards: triple
        - >= 10 cards: double
        - Never played: 39 points
        - Others: remaining count
        """
        players = [ChuDaDiPlayer(pid) for pid in range(4)]
        players[0].set_current_hand([])  # winner
        players[1].set_current_hand(
            self._make_cards(
                ["2", "2", "A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4"]
            )  # 13 cards -> 13 * 3 = 39
        )
        players[2].set_current_hand(
            self._make_cards(["2", "A", "K", "Q", "J", "T", "9", "8"])  # 8 cards -> 8
        )
        players[3].set_current_hand(self._make_cards(["A", "K", "Q"]))  # 3 cards -> 3

        # Mark players as having played cards (not "never played")
        for i in range(1, 4):
            players[i].played_cards = [Card("D", "3")]

        payoffs = ChuDaDiJudger(None, northern_rule=True).judge_payoffs(players, winner_id=0)
        self.assertEqual(payoffs[0], 0)   # winner
        self.assertEqual(payoffs[1], -39)  # 13 cards * 3
        self.assertEqual(payoffs[2], -8)   # 8 cards
        self.assertEqual(payoffs[3], -3)   # 3 cards

    def test_scoring_never_played(self):
        """Test northern rule: never played = 39 points."""
        players = [ChuDaDiPlayer(pid) for pid in range(4)]
        players[0].set_current_hand([])  # winner
        players[1].set_current_hand([Card("D", "3")])  # 1 card but never played
        # played_cards is empty -> never played

        payoffs = ChuDaDiJudger(None, northern_rule=True).judge_payoffs(players, winner_id=0)
        self.assertEqual(payoffs[1], -39)  # never played = 39


if __name__ == "__main__":
    unittest.main()
