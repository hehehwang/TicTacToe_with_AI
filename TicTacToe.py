class tictactoe:
    def __init__(self):
        self.board = list('_'*9)
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
               
    def dispboard(self):
        # for i in range(3):
        #    print(' '.join(self.board[3*i:3*(i+1)]))
        disp_board = [f' {self.board[i]} ' if self.board[i] != '_' else f'({str(i)})' for i in range(9)]
        for i in range(3):
            print(' '.join(disp_board[3*i:3*(i+1)]))
                    
    def checkresult(self):
        winning_cases = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
        for wc in winning_cases:
            if self.board[wc[0]] != '_' and\
                    self.board[wc[0]] == self.board[wc[1]] and\
                    self.board[wc[1]] == self.board[wc[2]]:
                if self.board[wc[0]] == 'O': self.result = 1
                else: self.result = 2
        # for wc in winning_cases:
        #     if sum([self.board[i] == 'O' for i in wc]) == 3:
        #         self.result = 1
        #     elif sum([self.board[i] == 'X' for i in wc]) == 3:
        #         self.result = 2
        if '_' not in self.board:
            self.result = 3
        
    def play(self):
        marker = 'O'
        if self.player == 2: marker = 'X'
        
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
        
    def change_turn(self):
        self.player = 3 - self.player
        
if __name__ == "__main__":
    t = tictactoe()
    while t.result == 0:
        t.dispboard()
        t.play()
        t.checkresult()
        t.change_turn()
    t.dispboard()
    if t.result == 3:
        print('Game Tied')
    else:
        print(f'Player {t.result} Won!')
