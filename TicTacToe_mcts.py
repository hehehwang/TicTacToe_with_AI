from TicTacToe import TTT
from copy import deepcopy
from math import log, sqrt
from random import choice as rndchoice
import time

winning_cases = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                 (0, 3, 6), (1, 4, 7), (2, 5, 8),
                 (0, 4, 8), (2, 4, 6)]


class Node:
    def __init__(self, s, par_node=None, pre_action=None):
        # Q : points obtained by simulations
        # N : number of simulations
        self.parent = par_node
        self.child = []
        self.q = 0
        self.n = 0
        self.pre_action = pre_action
        self.state = s
        self.player = MCTS.current_player(s)
        self.utc = float('inf')
        self.result = MCTS.is_terminal(s)

    def __repr__(self):
        ratio = self.q / (self.n + 1)
        l = [str(e) for e in (self.pre_action, ''.join(self.state), self.q, self.n, str(ratio)[:5], str(self.utc)[:5])]
        return ' '.join(l)

    def update(self, v):
        # v : 1 - p1(X) wins;
        #     2 - p2(O) wins;
        #     3 - tie;
        self.n += 1
        if v == 3:
            self.q += 0.5
        elif v == 3 - self.player:
            # IMPORTANT : points shoud go to opponent of current player
            self.q += 1


class MCTS:
    def __init__(self, s):
        self.root = Node(s)
        self.expansion(self.root)

    def mcts(self, mode='iteration', criteria=10000, new_board=None):
        if new_board is not None:
            self.__init__(new_board)
        if mode == 'iter':
            for _ in range(criteria):
                # FOR DEBUGGING
                # if _%5000 == 0:
                #     continue
                self._mcts_loop()
            return criteria
        elif mode == 'time':
            start_time = time.time()
            time_criteria = criteria/1000
            iii = 0
            while time.time() - start_time < time_criteria:
                self._mcts_loop()
                iii += 1
            return iii
        else:
            raise NotImplementedError

    def mcts_disp(self):
        # Search best child node of root node and print it
        best_node, best_visits = None, 0
        for n in self.root.child:
            print(n)
            if n.n > best_visits: best_visits, best_node = n.n, n
        print()
        print('Best Choice: ', best_node)
        print()

    def mcts_value_only(self):
        # Search best child node of root node and print it
        best_node, best_visits = None, 0
        for n in self.root.child:
            if n.n > best_visits: best_visits, best_node = n.n, n
        return best_node.pre_action

    def _mcts_loop(self):
        # One loop of MCTS
        # MCTS LOOP : selection -> expand -> simulation -> backpropagation
        node = self.selection(self.root)
        self.expansion(node)

        if node.child:
            selected_node = rndchoice(node.child)
        else:
            selected_node = node

        v = self.simulation(deepcopy(selected_node.state))
        self.backpropagation(selected_node, v)

    def selection(self, node):
        # Search the best node to simulate by UTC
        if node.child:
            imax, vmax = 0, 0
            for i, n in enumerate(node.child):
                n.utc = MCTS.utc(n)
                v = n.utc
                if v > vmax:
                    imax, vmax = i, v
            selected = node.child[imax]
            return self.selection(selected)
        else:
            selected = node
            return selected

    def expansion(self, node):
        # Append children node when the node is not terminal
        if self.is_terminal(node.state) == 0:
            actions = self.actions_available(node.state)
            for a in actions:
                state_after_action = self.action_result(node.state, a)
                node.child.append(Node(state_after_action, node, a))

    def simulation(self, s):
        # Do random playouts
        if self.is_terminal(s) == 0:
            actions = self.actions_available(s)
            a = rndchoice(actions)
            s = self.action_result(s, a)
            return self.simulation(s)
        else:
            return self.is_terminal(s)

    def backpropagation(self, node, v):
        node.update(v)
        if node.parent:
            self.backpropagation(node.parent, v)

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
    def actions_available(s):
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

    @staticmethod
    def utc(node):
        #                            ---------------------
        #                Q(v_i)     / 2 * ln(N(v_{i-1}))
        # UCT(v_i, v) = -------- + / --------------------
        #                N(v_i)   v        N(v_i)
        #
        v = node.q / (node.n + 1e-12) + sqrt(2 * log(node.parent.n + 1) / (node.n + 1e-12))
        return v


if __name__ == '__main__':
    t = TTT()
    m = MCTS(t.board)
    while t.result == 0:
        t.dispboard()
        ii = m.mcts(s=t.board, mode='time', criteria=50)
        print(ii)
        m.mcts_disp()
        t.player_input()
        t.checkresult()
        t.switch_player()
    t.dispboard()
    if t.result == 3:
        print('Game ended in DRAW')
    else:
        print(f'Player {t.result}({TTT.player_marker(t.result)}) Won!')
