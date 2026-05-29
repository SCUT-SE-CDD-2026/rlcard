from dataclasses import dataclass

from rlcard.games.base import Card

RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]
STRAIGHT_RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
SUIT_ORDER = ["D", "C", "H", "S"]

RANK_TO_VALUE = {rank: idx for idx, rank in enumerate(RANK_ORDER)}
STRAIGHT_RANK_TO_INDEX = {rank: idx for idx, rank in enumerate(STRAIGHT_RANK_ORDER)}
SUIT_TO_VALUE = {suit: idx for idx, suit in enumerate(SUIT_ORDER)}

# Feature/action encoding order: [D, C, H, S] x [3..K, A, 2]
DECK_SUIT_ORDER = ["D", "C", "H", "S"]
DECK_RANK_ORDER = ["3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A", "2"]

CARD_ID = {}
ID_TO_CARD = []
for suit_index, suit in enumerate(DECK_SUIT_ORDER):
    for rank_index, rank in enumerate(DECK_RANK_ORDER):
        card = Card(suit, rank)
        CARD_ID[(suit, rank)] = suit_index * len(DECK_RANK_ORDER) + rank_index
        ID_TO_CARD.append(card)

START_CARD = Card("D", "3")

# Production Android supports exactly eight non-pass combination types.
ACTION_TYPES = [
    "single",
    "pair",
    "triple",
    "straight",
    "flush",
    "full_house",
    "four_of_a_kind",  # 4+1, 铁支
    "straight_flush",
]

FIVE_CARD_TYPE_POWER = {
    "straight": 1,
    "flush": 2,
    "full_house": 3,
    "four_of_a_kind": 4,
    "straight_flush": 5,
}
BOMB_POWER = {
    "four_of_a_kind": 1,
    "straight_flush": 2,
}
ACTION_TYPE_PRIORITY = {name: idx for idx, name in enumerate(ACTION_TYPES)}
MASK_INDICES_CACHE = {}


def card_to_id(card):
    return CARD_ID[(card.suit, card.rank)]


def action_id_to_cards(action_id):
    if action_id == 0:
        return []
    cards = []
    for idx in range(len(ID_TO_CARD)):
        if action_id & (1 << idx):
            cards.append(ID_TO_CARD[idx])
    return cards


def cards_to_action_id(cards):
    action_id = 0
    for card in cards:
        action_id |= 1 << card_to_id(card)
    return action_id


def card_key(card):
    return (RANK_TO_VALUE[card.rank], SUIT_TO_VALUE[card.suit])


def sort_cards(cards):
    return sorted(cards, key=card_key)


def cards_to_str(cards, assume_sorted=False):
    if not cards:
        return "pass"
    if not assume_sorted:
        cards = sort_cards(cards)
    return " ".join(str(card) for card in cards)


@dataclass(frozen=True)
class Action:
    cards: tuple
    action_type: str
    length: int
    key: tuple
    raw: str

    def to_id(self):
        if self.action_type == "pass":
            return 0
        return cards_to_action_id(self.cards)


PASS_ACTION = Action(cards=tuple(), action_type="pass", length=0, key=tuple(), raw="pass")


def _is_straight(ranks):
    if "2" in ranks:
        return False
    indices = [STRAIGHT_RANK_TO_INDEX[rank] for rank in ranks]
    if len(set(indices)) != len(indices):
        return False
    indices.sort()
    return indices[-1] - indices[0] == len(indices) - 1


def _get_max_card(cards):
    return max(cards, key=card_key)


def _rank_counts(cards):
    counts = {}
    for card in cards:
        counts[card.rank] = counts.get(card.rank, 0) + 1
    return counts


def make_action(cards):
    if not cards:
        return None
    cards = sort_cards(cards)
    if len({(card.suit, card.rank) for card in cards}) != len(cards):
        return None

    length = len(cards)
    ranks = [card.rank for card in cards]
    suits = [card.suit for card in cards]
    rank_counts = _rank_counts(cards)
    unique_ranks = len(rank_counts)
    is_flush = all(suit == suits[0] for suit in suits)
    is_straight = _is_straight(ranks) if length == 5 else False

    if length == 1:
        max_card = cards[-1]
        return Action(tuple(cards), "single", length, card_key(max_card), cards_to_str(cards, True))

    if length == 2 and unique_ranks == 1:
        rank_value = RANK_TO_VALUE[ranks[0]]
        max_suit = SUIT_TO_VALUE[cards[-1].suit]
        return Action(tuple(cards), "pair", length, (rank_value, max_suit), cards_to_str(cards, True))

    if length == 3 and unique_ranks == 1:
        rank_value = RANK_TO_VALUE[ranks[0]]
        max_suit = SUIT_TO_VALUE[cards[-1].suit]
        return Action(tuple(cards), "triple", length, (rank_value, max_suit), cards_to_str(cards, True))

    # Production rules do not allow a naked four-card bomb.
    if length != 5:
        return None

    if is_flush and is_straight:
        max_card = cards[-1]
        return Action(tuple(cards), "straight_flush", length, card_key(max_card), cards_to_str(cards, True))

    if sorted(rank_counts.values()) == [1, 4]:
        quad_rank = next(rank for rank, count in rank_counts.items() if count == 4)
        quad_suit = max(SUIT_TO_VALUE[card.suit] for card in cards if card.rank == quad_rank)
        return Action(
            tuple(cards),
            "four_of_a_kind",
            length,
            (RANK_TO_VALUE[quad_rank], quad_suit),
            cards_to_str(cards, True),
        )

    if sorted(rank_counts.values()) == [2, 3]:
        triple_rank = next(rank for rank, count in rank_counts.items() if count == 3)
        triple_suit = max(SUIT_TO_VALUE[card.suit] for card in cards if card.rank == triple_rank)
        return Action(
            tuple(cards),
            "full_house",
            length,
            (RANK_TO_VALUE[triple_rank], triple_suit),
            cards_to_str(cards, True),
        )

    if is_flush:
        max_card = cards[-1]
        return Action(tuple(cards), "flush", length, card_key(max_card), cards_to_str(cards, True))

    if is_straight:
        max_card = cards[-1]
        return Action(tuple(cards), "straight", length, card_key(max_card), cards_to_str(cards, True))

    return None


def action_to_feature_meta(cards):
    if not cards:
        return "none", None, None
    action = make_action(cards)
    if action is None:
        return "none", None, None

    action_type = action.action_type
    main_rank = None
    kicker_rank = None

    if action_type in ("single", "pair", "triple"):
        main_rank = action.cards[0].rank
    elif action_type in ("straight", "flush", "straight_flush"):
        max_card = _get_max_card(action.cards)
        main_rank = max_card.rank
        if len(action.cards) >= 2:
            kicker_rank = action.cards[-2].rank
    elif action_type == "full_house":
        rank_counts = _rank_counts(action.cards)
        for rank, count in rank_counts.items():
            if count == 3:
                main_rank = rank
            elif count == 2:
                kicker_rank = rank
    elif action_type == "four_of_a_kind":
        rank_counts = _rank_counts(action.cards)
        for rank, count in rank_counts.items():
            if count == 4:
                main_rank = rank
            elif count == 1:
                kicker_rank = rank

    main_index = RANK_TO_VALUE[main_rank] if main_rank is not None else None
    kicker_index = RANK_TO_VALUE[kicker_rank] if kicker_rank is not None else None
    return action_type, main_index, kicker_index


def _compare_same_type(action, last_action):
    return action.key > last_action.key


def _can_beat_northern(action, last_action):
    if action.length != last_action.length:
        return False
    if action.action_type == last_action.action_type:
        return _compare_same_type(action, last_action)
    if action.length == 5:
        return FIVE_CARD_TYPE_POWER[action.action_type] > FIVE_CARD_TYPE_POWER[last_action.action_type]
    return False


def _can_beat_southern(action, last_action):
    action_is_bomb = action.action_type in BOMB_POWER
    last_is_bomb = last_action.action_type in BOMB_POWER
    if action_is_bomb or last_is_bomb:
        if action_is_bomb and last_is_bomb:
            return BOMB_POWER[action.action_type] > BOMB_POWER[last_action.action_type]
        return action_is_bomb
    if action.length != last_action.length or action.action_type != last_action.action_type:
        return False
    return _compare_same_type(action, last_action)


def can_beat(action, last_action, northern_rule=True):
    if last_action is None:
        return True
    if northern_rule:
        return _can_beat_northern(action, last_action)
    return _can_beat_southern(action, last_action)


def _get_mask_indices(num_cards):
    cache = MASK_INDICES_CACHE.get(num_cards)
    if cache is not None:
        return cache
    cache = [[] for _ in range(1 << num_cards)]
    for mask in range(1, 1 << num_cards):
        lsb = mask & -mask
        idx = lsb.bit_length() - 1
        prev = mask ^ lsb
        cache[mask] = cache[prev] + [idx]
    MASK_INDICES_CACHE[num_cards] = cache
    return cache


def _generate_valid_actions(hand):
    cards = sort_cards(hand)
    num_cards = len(cards)
    actions = []
    mask_indices = _get_mask_indices(num_cards)
    for mask in range(1, 1 << num_cards):
        indices = mask_indices[mask]
        length = len(indices)
        if length not in (1, 2, 3, 5):
            continue
        subset = [cards[i] for i in indices]
        action = make_action(subset)
        if action is not None:
            actions.append(action)
    return actions


def get_legal_actions(hand, last_action, must_contain_card=False, northern_rule=True):
    """Return production-rule legal actions for the current hand."""
    actions = _generate_valid_actions(hand)
    if must_contain_card:
        actions = [action for action in actions if START_CARD in action.cards]

    if last_action is not None:
        beatable_actions = [
            action for action in actions if can_beat(action, last_action, northern_rule)
        ]
        if northern_rule:
            actions = beatable_actions if beatable_actions else [PASS_ACTION]
        else:
            actions = beatable_actions + [PASS_ACTION] if beatable_actions else [PASS_ACTION]

    actions.sort(
        key=lambda action: (
            ACTION_TYPE_PRIORITY.get(action.action_type, 99),
            action.length,
            action.key,
            [card_key(card) for card in action.cards],
        )
    )
    return actions
