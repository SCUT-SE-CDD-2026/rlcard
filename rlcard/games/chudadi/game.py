import numpy as np

from rlcard.games.chudadi.dealer import ChuDaDiDealer as Dealer
from rlcard.games.chudadi.judger import ChuDaDiJudger as Judger
from rlcard.games.chudadi.player import ChuDaDiPlayer as Player
from rlcard.games.chudadi.round import ChuDaDiRound as Round


class ChuDaDiGame:
    def __init__(self, allow_step_back=False, northern_rule=True):
        self.allow_step_back = allow_step_back
        self.northern_rule = northern_rule
        self.np_random = np.random.RandomState()
        self.num_players = 4
        self.state = None
        self.winner_id = None

    def init_game(self):
        self.winner_id = None
        self.history = []
        self.players = [Player(player_id) for player_id in range(self.num_players)]
        self.dealer = Dealer(self.np_random)
        self.round = Round(self.np_random, self.num_players, self.northern_rule)
        self.round.initiate(self.players, self.dealer)
        self.judger = Judger(self.np_random, self.northern_rule)
        player_id = self.round.current_player
        self.state = self.get_state(player_id)
        return self.state, player_id

    def step(self, action):
        if self.allow_step_back:
            pass

        player = self.players[self.round.current_player]
        self.round.proceed_round(player, action)

        if len(player.current_hand) == 0:
            self.winner_id = player.player_id

        next_id = self.round.current_player
        state = self.get_state(next_id)
        self.state = state
        return state, next_id

    def step_back(self):
        return False

    def get_state(self, player_id):
        player = self.players[player_id]
        if self.is_over():
            action_ids = []
            raw_actions = []
        else:
            must_contain_start = (
                self.round.is_first_trick and player_id == self.round.starting_player
            )
            actions = self.judger.get_legal_actions(
                player.current_hand, self.round.last_action, must_contain_start
            )
            action_ids = [action.to_id() for action in actions]
            raw_actions = [action.raw for action in actions]
        last_action_cards = []
        last_action_type = None
        last_action_length = 0
        if self.round.last_action is not None:
            last_action_cards = list(self.round.last_action.cards)
            last_action_type = self.round.last_action.action_type
            last_action_length = self.round.last_action.length

        state = {
            "current_player": player_id,
            "current_hand": list(player.current_hand),
            "last_action": last_action_cards,
            "last_action_type": last_action_type,
            "last_action_length": last_action_length,
            "num_cards_left": [len(p.current_hand) for p in self.players],
            "pass_count": self.round.pass_count,
            "last_player": self.round.last_player,
            "is_first_trick": self.round.is_first_trick,
            "played_cards": [list(p.played_cards) for p in self.players],
            "actions": action_ids,
            "raw_legal_actions": raw_actions,
            "trace": list(self.round.trace),
            "northern_rule": self.northern_rule,
        }
        return state

    def get_payoffs(self):
        return self.judger.judge_payoffs(self.players, self.winner_id, self.northern_rule)

    def get_player_id(self):
        return self.round.current_player

    def get_num_players(self):
        return self.num_players

    @staticmethod
    def get_num_actions():
        return 1 << 52

    def is_over(self):
        return self.winner_id is not None
