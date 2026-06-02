from rlcard.games.chudadi.utils import START_CARD, card_key, can_beat, make_action


class ChuDaDiRound:
    def __init__(self, np_random, num_players, northern_rule=True, enable_baopei=True):
        self.np_random = np_random
        self.num_players = num_players
        self.northern_rule = northern_rule
        self.enable_baopei = enable_baopei
        self.current_player = 0
        self.starting_player = 0
        self.last_action = None
        self.last_player = None
        self.pass_count = 0
        self.is_first_trick = True
        self.trace = []
        self.played_action_history = [[] for _ in range(num_players)]
        self.baopei_player = None
        self.winning_action = None

    def initiate(self, players, dealer):
        dealer.shuffle()
        dealer.deal_cards(players)
        for player in players:
            if START_CARD in player.current_hand:
                self.starting_player = player.player_id
                break
        self.current_player = self.starting_player
        self.last_action = None
        self.last_player = None
        self.pass_count = 0
        self.is_first_trick = True
        self.trace = []
        self.played_action_history = [[] for _ in range(self.num_players)]
        self.baopei_player = None
        self.winning_action = None

    def proceed_round(self, player, action, players=None):
        if action == "pass" or action == []:
            self.trace.append((player.player_id, "pass"))
            self.pass_count += 1
            if self.pass_count >= self.num_players - 1 and self.last_player is not None:
                self.current_player = self.last_player
                self.last_action = None
                self.pass_count = 0
            else:
                self.current_player = (player.player_id + 1) % self.num_players
            return

        action_obj = make_action(action)
        if action_obj is None:
            raise ValueError(f"Invalid ChuDaDi action: {action}")

        if self.enable_baopei and players is not None:
            self._check_baopei(players, player.player_id, action_obj)

        self.trace.append((player.player_id, action_obj.raw))
        self.played_action_history[player.player_id].append(tuple(action_obj.cards))
        self.last_action = action_obj
        self.last_player = player.player_id
        self.pass_count = 0
        self.is_first_trick = False
        player.play_cards(list(action_obj.cards))
        self.winning_action = action_obj
        self.current_player = (player.player_id + 1) % self.num_players

    def _check_baopei(self, players, player_id, action_obj):
        # Production Baopei applies only to a single response when the next active
        # player has one card, and the responder could have played a stronger single
        # but did not. This mirrors BaopeiChecker's observable scoring effect.
        if action_obj.action_type != "single" or self.last_action is None:
            return
        if self.last_action.action_type != "single":
            return

        next_id = (player_id + 1) % self.num_players
        next_player = players[next_id]
        if len(next_player.current_hand) != 1:
            return

        next_single = make_action(next_player.current_hand)
        if next_single is not None and not can_beat(next_single, action_obj, self.northern_rule):
            return

        remaining = [card for card in players[player_id].current_hand if card not in action_obj.cards]
        if not remaining:
            return
        max_remaining = max(remaining, key=card_key)
        if card_key(action_obj.cards[0]) < card_key(max_remaining):
            self.baopei_player = player_id
