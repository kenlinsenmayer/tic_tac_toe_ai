import random

def blankboard():
    return list('bbbbbbbbb')

def move(brd, pos, piece):
    brd[pos] = piece
    return brd
    
def get_blanks(brd):
    return [i for i, v in enumerate(brd) if v == 'b']
    
def random_move(brd, piece):
    open_spaces = get_blanks(brd)
    return move(brd, random.choice(open_spaces), piece)

def isWinner(brd, piece):
    win_posns = [[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8],
                 [0, 3, 6],
                 [1, 4, 7],
                 [2, 5, 8],
                 [0, 4, 8],
                 [2, 4, 6]]
    def map_posns(xs):
        return [brd[i] for i in xs]
    piece3 = [piece, piece, piece]
    return piece3 in [map_posns(x) for x in win_posns]
    
def status(brd):
    if isWinner(brd, 'x'):
        return 'xwins'
    elif isWinner(brd, 'o'):
        return 'owins'
    elif len(get_blanks(brd)) == 0:
        return 'draw'
    else:
        return 'continue'
    
def play_game():
    brd = blankboard()
    next_move = 'x'
    move_list = []
    while True:
        move_list.append(tuple(brd))
        st = status(brd)
        if st == 'draw':
            return (move_list, 0.5)
        elif st == 'xwins':
            return (move_list, 1.0)
        elif st == 'owins':
            return (move_list, 0.0)
        else:
            brd = random_move(brd, next_move)
            next_move = 'o' if next_move == 'x' else 'x'

def mean(xs):
    return sum(xs) / len(xs)
    
def update_pct(x, target, pct):
    dx = target - x
    return x + (pct * dx)
    
def update_dict_from_list(d, brd_list, target, pct):
    for brd in brd_list:
        cur_val = d.get(brd, 0.5)
        new_val = update_pct(cur_val, target, pct)
        d[brd] = new_val
    return d
    
def play_a_random_game(d, pct):
    xs, target = play_game()
    update_dict_from_list(d, xs, target, pct)
    return d
    
def play_n_random_games(n, d, pct):
    for _ in range(n):
        play_a_random_game(d, pct)
    return d
    
learned_vals = play_n_random_games(100000, {}, 0.05)

#
# vals are learned, now time to play
#
from collections import Counter

def pick_best_x(d, brd):
    open_spaces = get_blanks(brd)
    def eval_pos(n):
        new_brd = move(brd.copy(), n, 'x')
        return d[tuple(new_brd)]
    best_move = max(open_spaces, key=eval_pos)
    return move(brd, best_move, 'x')

def play_game_for_smart_x(d):
    brd = blankboard()
    next_move = 'x'
    move_list = []
    while True:
        move_list.append(tuple(brd))
        st = status(brd)
        if st == 'draw':
            return (move_list, st)
        elif st == 'xwins':
            return (move_list, st)
        elif st == 'owins':
            return (move_list, st)
        else:
            if next_move == 'x':
                brd = pick_best_x(d, brd)
            else:
                brd = random_move(brd, next_move)
            next_move = 'o' if next_move == 'x' else 'x'

def play_n_smart_x_games(n, d):
    res = (play_game_for_smart_x(d)[1] for _ in range(n))
    return Counter(res)
    
print(play_n_smart_x_games(1000, learned_vals))