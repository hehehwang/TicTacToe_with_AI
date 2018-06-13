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

def auto_mcts(board, mode, criteria):
    tmp = MCTS(board)
    iii = tmp.mcts(mode=mode, criteria=criteria)
    return iii, tmp.mcts_value_only()


mmm, rrr = 20, 25 # Match, Round
mcts_criteria = ['time', 3] # ['time' or 'iter', millisecond or iteration cycle]
plt_fontsize = {'title':'small', 'legend':'x-small', 'x_lbl':'small', 'y_lbl':'small', 'tick_lbl':'small'}
fig_dpi = 300 # size of output img

look_through = False

fig, axes = plt.subplots(2,2,dpi=fig_dpi)

# 1. Random vs Random ===============================================
p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

mcts_iii = np.zeros(rrr)
mcts_iterations = np.zeros(mmm)

time_start = time.time()
if look_through:
    for i in range(mmm):
        for j in range(rrr):
            t = TTT()
            t.dispboard()
            print()
            while t.result == 0:
                if t.player == 1:
                    v = auto_random(t.board)
                    t.ai_input(v)
                else:
                    v = auto_random(t.board)
                    t.ai_input(v)
                t.dispboard()
                print()
                t.checkresult()
                t.switch_player()
            r = t.result
            print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1
        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

else:
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
                    v = auto_random(t.board)
                    t.ai_input(v)
                # t.dispboard()
                # print()
                t.checkresult()
                t.switch_player()
            r = t.result
            # print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1
        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)


time_elapsd = round(time.time()-time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

ax0 = axes[0,0]
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = RANDOM')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = RANDOM')
ax0.plot(iterations, tie_data, 'g--', lw=2, label='Tie')
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax0.set_ylim(-1, 101)
title_txt = 'Random(P1, X) vs Ransmall, O)'
# title_txt += f'\nP1:{p1_data[-1]}% P2: {p2_data[-1]}% elapsed: {time_elapsd}'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'])
ax0.legend(fontsize=plt_fontsize['legend'])


# 2. Random vs MCTS ===============================================


p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

mcts_iii = np.zeros(rrr)
mcts_iterations = np.zeros(mmm)

time_start = time.time()
if look_through:
    for i in range(mmm):
        for j in range(rrr):
            t = TTT()
            t.dispboard()
            print()
            while t.result == 0:
                if t.player == 1:
                    v = auto_random(t.board)
                    t.ai_input(v)
                else:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    mcts_iii[j] = v[0]
                    t.ai_input(v[1])
                t.dispboard()
                print()
                t.checkresult()
                t.switch_player()
            r = t.result
            print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1
        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

        mcts_iterations[i] = np.mean(mcts_iii)
        mcts_iii = np.zeros(rrr)
else:
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
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    mcts_iii[j] = v[0]
                    t.ai_input(v[1])
                # t.dispboard()
                # print()
                t.checkresult()
                t.switch_player()
            r = t.result
            # print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1

        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

        mcts_iterations[i] = np.mean(mcts_iii)
        mcts_iii = np.zeros(rrr)

time_elapsd = round(time.time()-time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()


ax0 = axes[1,0]
ax1 = ax0.twinx()
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = RANDOM')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = MCTS')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, mcts_iterations, 'c--', lw=2, label='MCTS Iterations')
ax1.plot(iterations,[np.mean(mcts_iterations)]*len(iterations), 'c-.', lw=1)
ax0.legend(loc=6, fontsize=plt_fontsize['legend'])
ax1.legend(loc=7, fontsize=plt_fontsize['legend'])
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax1.set_ylabel('Iterations', fontsize=plt_fontsize['y_lbl'])
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax1.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax0.set_ylim(-1, 101)
title_txt = 'Random(P1, X) vs MCTS(P2, O)'
if mcts_criteria[0] == 'time':
    title_txt += '\nTime limit: ' + str(mcts_criteria[1])+'ms'
else:
    title_txt += '\nCycles limit: ' + str(mcts_criteria[1])+'(Cycles)'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'])


# 3. MCTS vs Random ===============================================

p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

mcts_iii = np.zeros(rrr)
mcts_iterations = np.zeros(mmm)

time_start = time.time()
if look_through:
    for i in range(mmm):
        for j in range(rrr):
            t = TTT()
            t.dispboard()
            print()
            while t.result == 0:
                if t.player == 2:
                    v = auto_random(t.board)
                    t.ai_input(v)
                else:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    mcts_iii[j] = v[0]
                    t.ai_input(v[1])
                t.dispboard()
                print()
                t.checkresult()
                t.switch_player()
            r = t.result
            print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1
        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

        mcts_iterations[i] = np.mean(mcts_iii)
        mcts_iii = np.zeros(rrr)

else:
    for i in range(mmm):
        for j in range(rrr):
            t = TTT()
            # t.dispboard()
            # print()
            while t.result == 0:
                if t.player == 2:
                    v = auto_random(t.board)
                    t.ai_input(v)
                else:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    mcts_iii[j] = v[0]
                    t.ai_input(v[1])
                # t.dispboard()
                # print()
                t.checkresult()
                t.switch_player()
            r = t.result
            # print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1

        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

        mcts_iterations[i] = np.mean(mcts_iii)
        mcts_iii = np.zeros(rrr)

time_elapsd = round(time.time()-time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

ax0 = axes[1,1]
ax1 = ax0.twinx()
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = MCTS')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = RANDOM')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, mcts_iterations, 'm--', lw=2, label='MCTS Iterations')
ax1.plot(iterations,[np.mean(mcts_iterations)]*len(iterations), 'm-.', lw=1)
ax0.legend(loc=6, fontsize=plt_fontsize['legend'])
ax1.legend(loc=7, fontsize=plt_fontsize['legend'])
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax1.set_ylabel('Iterations', fontsize=plt_fontsize['y_lbl'])
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax1.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax0.set_ylim(-1, 101)

title_txt = 'MCTS(P1, X) vs Random(P2, O)'
if mcts_criteria[0] == 'time':
    title_txt += '\nTime limit: ' + str(mcts_criteria[1])+'ms'
else:
    title_txt += '\nCycles limit: ' + str(mcts_criteria[1])+'(Cycles)'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'])


# 4. MCTS vs MCTS ===============================================

p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

p1_iii, p2_iii = np.zeros(rrr), np.zeros(rrr)
p1_iterations, p2_iterations = np.zeros(mmm), np.zeros(mmm)

time_start = time.time()

if look_through:
    for i in range(mmm):
        for j in range(rrr):
            t = TTT()
            t.dispboard()
            print()
            while t.result == 0:
                if t.player == 2:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    p1_iii[j] = v[0]
                    t.ai_input(v[1])
                else:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    p2_iii[j] = v[0]
                    t.ai_input(v[1])
                t.dispboard()
                print()
                t.checkresult()
                t.switch_player()
            r = t.result
            print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1
        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

        p1_iterations[i], p2_iterations[i] = np.mean(p1_iii), np.mean(p2_iii)
        p1_iii, p2_iii = np.zeros(rrr), np.zeros(rrr)

else:
    for i in range(mmm):
        for j in range(rrr):
            t = TTT()
            # t.dispboard()
            # print()
            while t.result == 0:
                if t.player == 1:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    p1_iii[j] = v[0]
                    t.ai_input(v[1])
                else:
                    v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                    p2_iii[j] = v[0]
                    t.ai_input(v[1])
                # t.dispboard()
                # print()
                t.checkresult()
                t.switch_player()
            r = t.result
            # print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
            if r == 1:
                p1_wins += 1
            elif r == 2:
                p2_wins += 1
            else:
                tie += 1

        p1_data.append(p1_wins / iterations[i] * 100)
        p2_data.append(p2_wins / iterations[i] * 100)
        tie_data.append(tie / iterations[i] * 100)

        p1_iterations[i], p2_iterations[i] = np.mean(p1_iii), np.mean(p2_iii)
        p1_iii, p2_iii = np.zeros(rrr), np.zeros(rrr)

time_elapsd = round(time.time()-time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

ax0 = axes[0,1]
ax1 = ax0.twinx()

ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = MCTS')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = MCTS')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, p1_iterations, 'm--', lw=2, label='P1 MCTS Iterations')
ax1.plot(iterations, p2_iterations, 'c--', lw=2, label='P2 MCTS Iterations')
ax1.plot(iterations,[np.mean(p1_iterations)]*len(iterations), 'm-.', lw=1)
ax1.plot(iterations,[np.mean(p2_iterations)]*len(iterations), 'c-.', lw=1)
ax0.legend(loc=1, fontsize=plt_fontsize['legend'])
ax1.legend(loc=2, fontsize=plt_fontsize['legend'])
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax1.set_ylabel('Iterations', fontsize=plt_fontsize['y_lbl'])
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax1.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax0.set_ylim(-1, 101)

title_txt = 'MCTS(P1, X) vs MCTS(P2, O)'
if mcts_criteria[0] == 'time':
    title_txt += '\n Time limit: ' + str(mcts_criteria[1])+'ms'
else:
    title_txt += '\n Cycles limit: ' + str(mcts_criteria[1])+'(Cycles)'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'])


plt.tight_layout()
plt.show()