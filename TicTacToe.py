# Tic Tac Toe for python
# by hehehwang@gmail.com
# - 2018.05.18

class TTT:
    def __init__(self):
        self.board = list('_' * 9)
        # Board Index:
        # 012
        # 345
        # 678

        self.result = 0
        # 0: nothing
        # 1: p1 wins
        # 2: p2 wins
        # 3: tie

        self.player = 1
        # player1's marker = X
        # player2's marker = O

    def dispboard(self):
        # adding spaces b/w board markers and showing player the index of the board
        disp_board = [f' {self.board[i]} ' if self.board[i] != '_' else f'({str(i)})' for i in range(9)]
        for i in range(3):
            print(' '.join(disp_board[3 * i:3 * (i + 1)]))
        # LEGACY CODE; changed for better interface
        # for i in range(3):
        #    print(' '.join(self.board[3*i:3*(i+1)]))

    def checkresult(self):
        winning_cases = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                         (0, 3, 6), (1, 4, 7), (2, 5, 8),
                         (0, 4, 8), (2, 4, 6)]

        for wc in winning_cases:
            if self.board[wc[0]] != '_' and \
                    self.board[wc[0]] == self.board[wc[1]] and \
                    self.board[wc[1]] == self.board[wc[2]]:
                if self.board[wc[0]] == 'X':
                    self.result = 1
                else:
                    self.result = 2
        # LEGACY CODE; changed for better performance (needs 1/2 times less searching)
        # for wc in winning_cases:
        #     if sum([self.board[i] == 'O' for i in wc]) == 3:
        #         self.result = 1
        #     elif sum([self.board[i] == 'X' for i in wc]) == 3:
        #         self.result = 2

        if '_' not in self.board:
            self.result = 3

    def checkresult_with_returns(self, s=None):
        s = self.board if s is None else s

        winning_cases = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
                         (0, 3, 6), (1, 4, 7), (2, 5, 8),
                         (0, 4, 8), (2, 4, 6)]
        for wc in winning_cases:
            if s[wc[0]] != '_' and \
                    s[wc[0]] == s[wc[1]] and \
                    s[wc[1]] == s[wc[2]]:
                if s[wc[0]] == 'O':
                    return 1
                else:
                    return 2
        if '_' not in self.board:
            return 3
        else:
            return 0

    @staticmethod
    def player_marker(p):
        if p == 1:
            return 'X'
        else:
            return 'O'

    def player_input(self):
        marker = TTT.player_marker(self.player)
        input_msg = f'Player {self.player}({marker}), select your next position\n'

        while 1:
            try:
                p = int(input(input_msg))
                if self.board[p] != '_':
                    raise NotImplementedError
                else:
                    break
            except NotImplementedError:
                print("You can't place marker on that place")

        self.board[p] = marker
        print()

    def switch_player(self):
        self.player = 3 - self.player

    def play(self):
        while self.result == 0:
            self.dispboard()
            self.player_input()
            self.checkresult()
            self.switch_player()
        self.dispboard()
        if self.result == 3:
            print('Game ended in DRAW')
        else:
            print(f'Player {self.result}({TTT.player_marker(self.result)}) Won!')


if __name__ == "__main__":
    t = TTT()
    t.play()
