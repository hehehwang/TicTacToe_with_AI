from TicTacToe import tictactoe as ttt
from copy import deepcopy

winning_cases = [(0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)]
        
def current_player(s):
    n = s.count('_')
    if n%2 == 1: return 1
    return 2

def available_action(s):
    l = []
    for i in range(9):
        if s[i] == '_' : l.append(i)
    return l
    
def terminal(s):
    for wc in winning_cases:
        if s[wc[0]] != '_' and\
                s[wc[0]] == s[wc[1]] and\
                s[wc[1]] == s[wc[2]]:
                    return True
    if '_' not in s:
        return True
    return False

def state_evaluation(s, p):
    p = 'X' if p == 1 else 'O'
    for wc in winning_cases:
        if s[wc[0]] != '_' and\
                s[wc[0]] == s[wc[1]] and\
                s[wc[1]] == s[wc[2]]:
            if s[wc[0]] == p: return 10
            else: return -10
    if '_' not in s:
        return 0
        
def action_result(s, p, a):
    new_s = deepcopy(s)
    new_s[a] = 'X' if p == 1 else 'O'
    return new_s

def _function_test():
    t = ttt()
    while t.result == 0:
        t.dispboard()
        t.player_input()
        t.checkresult_old()
        t.change_turn()
        for f in [current_player, available_action, terminal, det_next_step]:
            print(str(f), f(t.board))
        print(state_evaluation, state_evaluation(t.board, t.player))
    t.dispboard()
    if t.result == 3:
        print('Game Tied')
    else:
        print(f'Player {t.result} Won!')

def minimax(s, p, deph = 0):
    if terminal(s): return state_evaluation(s,p)
    cp = current_player(s)
    val = -100 if cp == p else 100
    actions = available_action(s)
    for a in actions:
        s_new = action_result(s, cp, a)
        v = minimax(s_new, p, deph+1)
        val = max(val, v) if cp == p else min(val, v)
    return val

def det_next_step(s):
    p = current_player(s)
    if s == list('_'*9): return 0
    elif '_' not in s: return None
    values = []
    actions = available_action(s)
    for a in actions:
        s_new = action_result(s, p, a)
        values.append(minimax(s_new, p))
    return actions[values.index(max(values))]

if __name__ == '__main__':
    _function_test()