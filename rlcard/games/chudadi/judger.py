class ChuDaDiJudger:
    def __init__(self, np_random, northern_rule=True):
        self.np_random = np_random
        self.northern_rule = northern_rule

    def get_legal_actions(self, hand, last_action, must_contain_card):
        from rlcard.games.chudadi.utils import get_legal_actions

        return get_legal_actions(hand, last_action, must_contain_card, self.northern_rule)

    def judge_payoffs(
        self,
        players,
        winner_id,
        northern_rule=True,
        baopei_player=None,
        winning_action=None,
    ):
        """Judge zero-sum payoffs using Android ScoreCalculator semantics."""
        if winner_id is None:
            return [0 for _ in players]

        has_spade_two_in_winning_play = False
        if winning_action is not None:
            has_spade_two_in_winning_play = any(
                card.rank == "2" and card.suit == "S" for card in winning_action.cards
            )

        penalties = []
        for player in players:
            if player.player_id == winner_id:
                penalties.append(0)
                continue
            remaining = len(player.current_hand)
            if northern_rule:
                penalty = 52 if remaining == 13 else remaining * 2 if remaining >= 10 else remaining
            else:
                penalty = remaining
                if any(card.rank == "2" for card in player.current_hand):
                    penalty *= 2
            if has_spade_two_in_winning_play:
                penalty *= 2
            penalties.append(penalty)

        total_penalty = sum(penalties)
        payoffs = []
        for player in players:
            if player.player_id == winner_id:
                payoffs.append(total_penalty)
            elif baopei_player is not None and baopei_player != winner_id:
                payoffs.append(-total_penalty if player.player_id == baopei_player else 0)
            else:
                payoffs.append(-penalties[player.player_id])
        return payoffs
