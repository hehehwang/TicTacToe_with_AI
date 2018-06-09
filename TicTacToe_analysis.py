import numpy as np
from TicTacToe import TTT
from TicTacToe_mcts import MCTS
from TicTacToe_minimax import det_next_action_alpha_beta as det_minimax
from TicTacToe_random import random_action as det_random
import matplotlib.pylab as plt
import time

def auto_random(board):
    return det_random(board)

def auto_minimax(board):
    return det_minimax(board)[0]

def auto_mcts(board, mcts_mode, mcts_criteria):
    tmp = MCTS(board)
    iii = tmp.mcts(mode=mcts_mode, criteria=mcts_criteria)
    return iii, tmp.mcts_value_only()


mmm, rrr = 50, 20

p1_data, p2_data = [], []
p1_wins, p2_wins = 0, 0
iterations = np.arange(rrr,mmm*rrr+1,rrr)

mcts_iii = np.zeros(rrr)
mcts_iterations = np.zeros(mmm)

time_start = time.time()
for i in range(mmm):
    for j in range(rrr):
        t = TTT()
        # t.dispboard()
        # print()
        while t.result == 0:
            if t.player == 1:
                v = auto_random(t.board)
                t.ai_input(v)
            else:
                v = auto_mcts(t.board, 'time', 5)
                mcts_iii[j] = v[0]
                t.ai_input(v[1])
            # t.dispboard()
            # print()
            t.checkresult()
            t.switch_player()
        r = t.result
        # print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
        if r == 1: p1_wins += 1
        elif r == 2: p2_wins += 1
    p1_data.append(p1_wins/iterations[i]*100)
    p2_data.append(p2_wins/iterations[i]*100)
    mcts_iterations[i] = np.mean(mcts_iii)
    mcts_iii = np.zeros(rrr)

print(time.time()-time_start)
print(p1_data)
print(p2_data)


# plt.plot(iterations, p1_data, 'r', label='player1(X) = RANDOM')
# plt.plot(iterations, p2_data, 'b', label='Player2(O) = MCTS')
# plt.plot(iterations, mcts_iterations, 'c--', label='MCTS Iterations (Mean)')
# plt.title('Random(p1, X) vs MCTS(p2, O)')
# plt.legend()
# plt.show()

fig, ax0 = plt.subplots()
ax1 = ax0.twinx()
ax0.plot(iterations, p1_data, 'r', label='player1(X) = RANDOM')
ax0.plot(iterations, p2_data, 'b', label='Player2(O) = MCTS')
ax1.plot(iterations, mcts_iterations, 'c--', label='MCTS Iterations\n(mean value)')
ax0.legend(loc=6)
ax1.legend(loc=7)
ax0.set_xlabel('TicTacToe Rounds')
ax0.set_ylabel('Cumulative win Rate (%)')
ax1.set_ylabel('Iterations')
ax0.set_ylim(0,100)
plt.title('Random(p1, X) vs MCTS(p2, O) - time: 5ms')
plt.show()
