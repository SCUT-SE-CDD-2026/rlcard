from rlcard.games.chudadi.utils import get_legal_actions


class ChuDaDiJudger:
    def __init__(self, np_random, northern_rule=True):
        self.np_random = np_random
        self.northern_rule = northern_rule

    def get_legal_actions(self, hand, last_action, must_contain_card):
        return get_legal_actions(hand, last_action, must_contain_card, self.northern_rule)

    def judge_payoffs(self, players, winner_id, northern_rule=True):
        """Judge payoffs for the game."""
        if winner_id is None:
            return [0 for _ in players]

        scores = []
        for player in players:
            remaining = len(player.current_hand)
            if player.player_id == winner_id:
                score = 0
            else:
                # Northern rule scoring
                if northern_rule:
                    if len(player.played_cards) == 0:
                        # Never played: 39 points
                        score = 39
                    elif remaining >= 13:
                        score = remaining * 3
                    elif remaining >= 10:
                        score = remaining * 2
                    else:
                        score = remaining
                else:
                    # (original)
                    if remaining == 13:
                        score = 39
                    elif remaining >= 8:
                        score = remaining * 2
                    else:
                        score = remaining
                    twos = sum(1 for card in player.current_hand if card.rank == "2")
                    score += twos
            scores.append(score)

        total = sum(scores)
        payoffs = []
        for player in players:
            if player.player_id == winner_id:
                payoffs.append(total)
            else:
                payoffs.append(-scores[player.player_id])
        return payoffs
