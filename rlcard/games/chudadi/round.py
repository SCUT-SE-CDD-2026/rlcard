from rlcard.games.chudadi.utils import START_CARD, make_action


class ChuDaDiRound:
    def __init__(self, np_random, num_players, northern_rule=True):
        self.np_random = np_random
        self.num_players = num_players
        self.northern_rule = northern_rule
        self.current_player = 0
        self.starting_player = 0
        self.last_action = None
        self.last_player = None
        self.pass_count = 0
        self.is_first_trick = True
        self.trace = []

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

    def proceed_round(self, player, action):
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
        self.trace.append((player.player_id, action_obj.raw))
        self.last_action = action_obj
        self.last_player = player.player_id
        self.pass_count = 0
        self.is_first_trick = False
        player.play_cards(list(action_obj.cards))
        self.current_player = (player.player_id + 1) % self.num_players
