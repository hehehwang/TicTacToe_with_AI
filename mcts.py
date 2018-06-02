from TicTacToe import TTT
from copy import deepcopy
from math import log, sqrt
from random import choice as rndchoice

winning_cases = [(0,1,2),(3,4,5),(6,7,8),
    (0,3,6),(1,4,7),(2,5,8),
    (0,4,8),(2,4,6)]
    
class Node:
    def __init__(self, s, par_node = None):
        self.parent = par_node
        self.child = []
        self.q = 0
        self.n = 0
        self.state = s
        self.player = MCTS.current_player(s)
        self.utc = float('inf')

    def __str__(self):
        return ''.join(self.state) + ' ' + str(self.q) + '/' + str(self.n) + ' ' + str(self.q//self.n) + ' ' + str(self.utc)

    def update(self, v):
        self.n += 1
        if v == 3:
            self.q += 0.5
        elif v == self.player:
            self.q += 1

        #                            ----------------
        #                Q(v_i)     / 2 * ln(N(V))
        # UCT(v_i, v) = -------- + / ---------------
        #                N(v_i)   v     N(v_i)

        if self.parent:
            self.utc = self.q / (self.n) + sqrt(2 * log(self.parent.n+1) / (self.n))

class MCTS:
    def __init__(self, s):
        self.root = Node(s)
        self.expansion(self.root)

    def expansion(self, node):
        if self.is_terminal(node.state) == 0:
            actions = self.available_actions(node.state)
            for a in actions:
                node.child.append(Node(self.action_result(node.state, a), node))

    @staticmethod
    def is_terminal(s):
        for wc in winning_cases:
            if s[wc[0]] != '_' and \
                    s[wc[0]] == s[wc[1]] and \
                    s[wc[1]] == s[wc[2]]:
                if s[wc[0]] == 'X':
                    return 1
                else:
                    return 2
        if '_' not in s:
            return 3
        else:
            return 0

    @staticmethod
    def available_actions(s):
        l = []
        for i in range(9):
            if s[i] == '_': l.append(i)
        return l

    @staticmethod
    def action_result(s, a):
        # s : State, p : player no.
        p = MCTS.current_player(s)
        new_s = deepcopy(s)
        new_s[a] = 'X' if p == 1 else 'O'
        return new_s

    @staticmethod
    def current_player(s):
        n = s.count('_')
        if n % 2 == 1: return 1
        return 2

    def playout(self, s):
        if self.is_terminal(s) == 0:
            actions = self.available_actions(s)
            a = rndchoice(actions)
            s = self.action_result(s, a)
            return self.playout(s)
        else:
            return self.is_terminal(s)

    # @staticmethod
    # def utc(node):
    # #                            ----------------
    # #                Q(v_i)     / 2 * ln(N(V))
    # # UCT(v_i, v) = -------- + / ---------------
    # #                N(v_i)   v     N(v_i)
    # #
    #     v = node.q / (node.n + 1e-12) + sqrt(2 * log(node.parent.n + 1) / (node.n + 1e-12))
    #     return v

    def selection(self, node):
        child_nods = node.child
        if child_nods:
            imax, vmax = 0, 0
            for i, n in enumerate(child_nods):
                v = n.utc
                if  v > vmax : imax, vmax = i, v
            return self.selection(child_nods[imax])
        else:
            return node

    def backpropagation(self, node, v):
        node.update(v)
        if node.parent:
            self.backpropagation(node.parent, v)

    def mcts(self, mode = 'iteration', criteria = 10000):
        # selection -> expand -> playout -> backpropagation
        if mode == 'iteration':
            for _ in range(criteria):
                node = self.selection(self.root)
                self.expansion(node)
                children = node.child
                if children:
                    selected = rndchoice(children)
                else:
                    selected = node
                v = self.playout(deepcopy(selected.state))
                self.backpropagation(selected, v)
            for n in self.root.child:
                print(n)

if __name__ == '__main__':
    t = TTT()
    while t.result == 0:
        t.dispboard()
        m = MCTS(t.board)
        m.mcts()
        t.player_input()
        t.checkresult()
        t.switch_player()
        print(t.board)
    t.dispboard()
    if t.result == 3:
        print('Game ended in DRAW')
    else:
        print(f'Player {t.result}({TTT.player_marker(t.result)}) Won!')
