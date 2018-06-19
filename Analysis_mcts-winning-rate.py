import time

import matplotlib.pylab as plt
import numpy as np

from TicTacToe import TTT
from AI_mcts import MCTS
from AI_minimax import det_next_action_alpha_beta as det_minimax
from AI_random import random_action as det_random


def auto_random(board):
    return det_random(board)


def auto_minimax(board):
    return det_minimax(board)[0]


def auto_mcts(board, mode, criteria):
    tmp = MCTS(board)
    iii = tmp.mcts(mode=mode, criteria=criteria)
    return iii, tmp.result_return()

def mean(l):
    return sum(l)/len(l)

mmm, rrr = 20, 10  # Match, Round
mcts_criteria = ['time', 100]  # ['time' or 'iter', millisecond or iteration cycle]
plt_fontsize = {'title': 'x-small',
                'legend': 5,
                'x_lbl': 'x-small',
                'y_lbl': 'x-small',
                'tick_lbl': 'xx-small'}
fig_dpi = 300  # size of output img
look_through = True

save_route = f'output/mcts_vs_random_{mcts_criteria[0]}_'
if mcts_criteria[0] == 'time':
    save_route += f'{mcts_criteria[1]}ms'
elif mcts_criteria[0] == 'iter':
    save_route += f'{mcts_criteria[1]}cycles'
else:
    raise NotImplementedError
save_route += time.strftime('_%m%d_%H%M%S', time.localtime(time.time()))

fig, axes = plt.subplots(2, 2, dpi=fig_dpi)

with open(save_route + '.txt', 'w') as f:
    f.write('=== Settings ===\n')
    f.write(f'- Matchs, Rounds: {mmm}, {rrr}\n')
    f.write(f'- Look Through: {str(look_through)}\n')
    f.write(f'- MCTS Setting: {mcts_criteria[0]}, {mcts_criteria[1]}\n\n')

# 1. Random vs Random ===============================================
p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

time_start = time.time()
for i in range(mmm):
    for j in range(rrr):
        t = TTT()
        if look_through:
            t.dispboard()
            print()
        while t.result == 0:
            if t.player == 1:
                v = auto_random(t.board)
                t.ai_input(v)
            else:
                v = auto_random(t.board)
                t.ai_input(v)
            if look_through:
                t.dispboard()
                print()
            t.checkresult()
            t.switch_player()
        r = t.result
        if look_through: print(f'\n{i+1}/{mmm}, {j+1}/{rrr}, {r}\n\n')
        if r == 1:
            p1_wins += 1
        elif r == 2:
            p2_wins += 1
        else:
            tie += 1
    p1_data.append(p1_wins / iterations[i] * 100)
    p2_data.append(p2_wins / iterations[i] * 100)
    tie_data.append(tie / iterations[i] * 100)

time_elapsd = round(time.time() - time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

with open(save_route + '.txt', 'a') as f:
    f.write('CASE Ⅰ. RANDOM VS RANDOM\n')
    f.write(f'- P1 final winning rate: {round(p1_data[-1],4)}%\n')
    f.write(f'- P2 final winning rate: {round(p2_data[-1],4)}%\n')
    f.write(f'- Elapsed Time: {time_elapsd}s\n\n')

ax0 = axes[0, 0]
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = RANDOM')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = RANDOM')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax0.set_xlim(rrr, mmm*rrr)
ax0.set_ylim(-1, 101)
title_txt = 'Random(P1, X) vs Random(P2, O)'
# title_txt += f'\nP1:{p1_data[-1]}% P2: {p2_data[-1]}% elapsed: {time_elapsd}'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'], weight='bold')
ax0.legend(fontsize=plt_fontsize['legend'])

if look_through: time.sleep(3)

# 2. Random vs MCTS ===============================================
p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

mcts_iter_game, mcts_iter_round = [], []
mcts_iter_match = []*mmm

time_start = time.time()
for i in range(mmm):
    for j in range(rrr):
        t = TTT()
        if look_through:
            t.dispboard()
            print()
        while t.result == 0:
            if t.player == 1:
                v = auto_random(t.board)
                t.ai_input(v)
            else:
                v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                mcts_iter_game.append(v[0])
                t.ai_input(v[1])
            if look_through:
                t.dispboard()
                print()
            t.checkresult()
            t.switch_player()
        r = t.result
        if look_through:
            print(f'\n{i+1}/{mmm}, {j+1}/{rrr}, {r}\n\n')
        if r == 1:
            p1_wins += 1
        elif r == 2:
            p2_wins += 1
        else:
            tie += 1

        iii_avg, iii_max, iii_min = mean(mcts_iter_game), mcts_iter_game[-1], mcts_iter_game[0] # average, min, max
        mcts_iter_round.append((iii_avg, iii_max, iii_min))
        mcts_iter_game = []

    p1_data.append(p1_wins / iterations[i] * 100)
    p2_data.append(p2_wins / iterations[i] * 100)
    tie_data.append(tie / iterations[i] * 100)

    ii_avg, ii_max, ii_min = mean([i[0] for i in mcts_iter_round]), mean([i[1] for i in mcts_iter_round]), mean([i[2] for i in mcts_iter_round])
    # ii_avg, ii_max, ii_min = [[mean(i[j]) for i in mcts_iter_round] for j in range(3)]
    mcts_iter_match.append((ii_avg, ii_max, ii_min))
    mcts_iter_round = []


time_elapsd = round(time.time() - time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

with open(save_route + '.txt', 'a') as f:
    f.write('CASE Ⅱ. RANDOM VS MCTS\n')
    f.write(f'- P1 final winning rate: {round(p1_data[-1],4)}%\n')
    f.write(f'- P2 final winning rate: {round(p2_data[-1],4)}%\n')
    f.write(f'- MCTS(P2) Iterations(average value): {round(float(mean([i[0] for i in mcts_iter_match])),4)}\n')
    f.write(f'- MCTS(P2) Iterations(min value): {round(float(mean([i[2] for i in mcts_iter_match])),4)}\n')
    f.write(f'- Elapsed Time: {time_elapsd}s\n\n')

# set axes and twinx
ax0 = axes[1, 0]
ax1 = ax0.twinx()
ax0.grid(True)

# set plot
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1=RANDOM')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2=MCTS')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, [i[0] for i in mcts_iter_match], 'c--', lw=2, label='Iterations(mean)')
# ax1.plot(iterations, [i[1] for i in mcts_iter_match], 'c:', lw=0.8)
ax1.plot(iterations, [i[2] for i in mcts_iter_match], 'c:', lw=0.8, label='Iterations(min)')
ax1.plot(iterations, [mean([i[0] for i in mcts_iter_match])] * len(iterations), 'c-', lw=1, label='Average Line')


# set legends
ax0.legend(loc=6, fontsize=plt_fontsize['legend'])
ax1.legend(loc=7, fontsize=plt_fontsize['legend'])

# set axis label
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax1.set_ylabel('Iterations', fontsize=plt_fontsize['y_lbl'])

# set tick label
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax1.tick_params(labelsize=plt_fontsize['tick_lbl'])

# set x, y range
ax0.set_xlim(rrr, mmm*rrr)
ax0.set_ylim(-1, 101)

# set plot title
title_txt = 'Random(P1, X) vs MCTS(P2, O)'
if mcts_criteria[0] == 'time':
    title_txt += '\nTime limit: ' + str(mcts_criteria[1]) + 'ms'
else:
    title_txt += '\nCycles limit: ' + str(mcts_criteria[1]) + '(Cycles)'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'], weight='bold')

if look_through: time.sleep(3)


# 3. MCTS vs Random ===============================================
p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

mcts_iter_game, mcts_iter_round = [], []
mcts_iter_match = []*mmm

time_start = time.time()
for i in range(mmm):
    for j in range(rrr):
        t = TTT()
        if look_through:
            t.dispboard()
            print()
        while t.result == 0:
            if t.player == 2:
                v = auto_random(t.board)
                t.ai_input(v)
            else:
                v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                mcts_iter_game.append(v[0])
                t.ai_input(v[1])
            if look_through:
                t.dispboard()
                print()
            t.checkresult()
            t.switch_player()
        r = t.result
        if look_through:
            print(f'\n{i+1}/{mmm}, {j+1}/{rrr}, {r}\n\n')
        if r == 1:
            p1_wins += 1
        elif r == 2:
            p2_wins += 1
        else:
            tie += 1

        iii_avg, iii_max, iii_min = mean(mcts_iter_game), mcts_iter_game[-1], mcts_iter_game[0] # average, min, max
        mcts_iter_round.append((iii_avg, iii_max, iii_min))
        mcts_iter_game = []

    p1_data.append(p1_wins / iterations[i] * 100)
    p2_data.append(p2_wins / iterations[i] * 100)
    tie_data.append(tie / iterations[i] * 100)

    ii_avg, ii_max, ii_min = mean([i[0] for i in mcts_iter_round]), mean([i[1] for i in mcts_iter_round]), mean([i[2] for i in mcts_iter_round])
    # ii_avg, ii_max, ii_min = [[mean(i[j]) for i in mcts_iter_round] for j in range(3)]
    mcts_iter_match.append((ii_avg, ii_max, ii_min))
    mcts_iter_round = []


time_elapsd = round(time.time() - time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

with open(save_route + '.txt', 'a') as f:
    f.write('CASE Ⅲ. MCTS VS Random\n')
    f.write(f'- P1 final winning rate: {round(p1_data[-1],4)}%\n')
    f.write(f'- P2 final winning rate: {round(p2_data[-1],4)}%\n')
    f.write(f'- MCTS(P2) Iterations(average value): {round(float(mean([i[0] for i in mcts_iter_match])),4)}\n')
    f.write(f'- MCTS(P2) Iterations(min value): {round(float(mean([i[2] for i in mcts_iter_match])),4)}\n')
    f.write(f'- Elapsed Time: {time_elapsd}s\n\n')

# set axes and twinx
ax0 = axes[1, 1]
ax1 = ax0.twinx()
ax0.grid(True)

# set plot
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1=MCTS')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2=RANDOM')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, [i[0] for i in mcts_iter_match], 'm--', lw=2, label='Iterations(mean)')
# ax1.plot(iterations, [i[1] for i in mcts_iter_match], 'c:', lw=0.8)
ax1.plot(iterations, [i[2] for i in mcts_iter_match], 'm:', lw=0.8, label='Iterations(min)')
ax1.plot(iterations, [mean([i[0] for i in mcts_iter_match])] * len(iterations), 'm-', lw=1, label='Average Line')

# set legends
ax0.legend(loc=6, fontsize=plt_fontsize['legend'])
ax1.legend(loc=7, fontsize=plt_fontsize['legend'])

# set axis label
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax1.set_ylabel('Iterations', fontsize=plt_fontsize['y_lbl'])

# set tick label
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax1.tick_params(labelsize=plt_fontsize['tick_lbl'])

# set x, y range
ax0.set_xlim(rrr, mmm*rrr)
ax0.set_ylim(-1, 101)

# set plot title
title_txt = 'MCTS(P1, X) vs Random(P2, O)'
if mcts_criteria[0] == 'time':
    title_txt += '\nTime limit: ' + str(mcts_criteria[1]) + 'ms'
else:
    title_txt += '\nCycles limit: ' + str(mcts_criteria[1]) + '(Cycles)'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'], weight='bold')

if look_through: time.sleep(3)



# 4. MCTS vs MCTS ===============================================
p1_data, p2_data, tie_data = [], [], []
p1_wins, p2_wins, tie = 0, 0, 0
iterations = np.arange(rrr, mmm * rrr + 1, rrr)

p1_iii, p2_iii = np.zeros(rrr), np.zeros(rrr)
p1_iter_game, p2_iter_game, p1_iter_round, p2_iter_round, p1_iter_match, p2_iter_match = [[] for _ in range(6)]

time_start = time.time()

for i in range(mmm):
    for j in range(rrr):
        t = TTT()
        if look_through:
            t.dispboard()
            print()
        while t.result == 0:
            if t.player == 1:
                v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                p1_iter_game.append(v[0])
                t.ai_input(v[1])
            else:
                v = auto_mcts(t.board, mcts_criteria[0], mcts_criteria[1])
                p2_iter_game.append(v[0])
                t.ai_input(v[1])
            if look_through:
                t.dispboard()
                print()
            t.checkresult()
            t.switch_player()
        r = t.result
        if look_through:
            print(f'\n{i+1}/{mmm}, {j+1}/{rrr}, {r}\n\n')
        if r == 1:
            p1_wins += 1
        elif r == 2:
            p2_wins += 1
        else:
            tie += 1

        iii_avg, iii_max, iii_min = mean(p1_iter_game), p1_iter_game[-1], p1_iter_game[0] # average, min, max
        p1_iter_round.append((iii_avg, iii_max, iii_min))
        iii_avg, iii_max, iii_min = mean(p2_iter_game), p2_iter_game[-1], p2_iter_game[0]  # average, min, max
        p2_iter_round.append((iii_avg, iii_max, iii_min))

        p1_iter_game, p2_iter_game = [], []

    p1_data.append(p1_wins / iterations[i] * 100)
    p2_data.append(p2_wins / iterations[i] * 100)
    tie_data.append(tie / iterations[i] * 100)

    ii_avg, ii_max, ii_min = mean([i[0] for i in p1_iter_round]), mean([i[1] for i in p1_iter_round]), mean([i[2] for i in p1_iter_round])
    p1_iter_match.append((ii_avg, ii_max, ii_min))
    ii_avg, ii_max, ii_min = mean([i[0] for i in p2_iter_round]), mean([i[1] for i in p2_iter_round]), mean([i[2] for i in p2_iter_round])
    p2_iter_match.append((ii_avg, ii_max, ii_min))

    p1_iter_round, p2_iter_round = [], []

time_elapsd = round(time.time() - time_start, 3)
print('Elapsed Time: ', time_elapsd)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

with open(save_route + '.txt', 'a') as f:
    f.write('CASE Ⅳ. MCTS VS MCTS\n')
    f.write(f'- P1 final winning rate: {round(p1_data[-1],4)}%\n')
    f.write(f'- P2 final winning rate: {round(p2_data[-1],4)}%\n')
    f.write(f'- MCTS(P1) Iterations(average value): {round(float(mean([i[0] for i in p1_iter_match])),4)}\n')
    f.write(f'- MCTS(P1) Iterations(min value): {round(float(mean([i[2] for i in p1_iter_match])),4)}\n')
    f.write(f'- MCTS(P2) Iterations(average value): {round(float(mean([i[0] for i in p2_iter_match])),4)}\n')
    f.write(f'- MCTS(P2) Iterations(min value): {round(float(mean([i[2] for i in p2_iter_match])),4)}\n')
    f.write(f'- Elapsed Time: {time_elapsd}s\n\n')
    f.write(time.ctime())

# set axes and twinx
ax0 = axes[0, 1]
ax1 = ax0.twinx()
ax0.grid(True)

# set plot
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1=MCTS')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2=MCTS')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')

ax1.plot(iterations, [i[0] for i in p1_iter_match], 'm--', lw=2, label='Iterations(P1)')
ax1.plot(iterations, [i[2] for i in p1_iter_match], 'm:', lw=0.8)
ax1.plot(iterations, [mean([i[0] for i in p1_iter_match])] * len(iterations), 'm-', lw=1)

ax1.plot(iterations, [i[0] for i in p2_iter_match], 'c--', lw=2, label='Iterations(P2)')
ax1.plot(iterations, [i[2] for i in p2_iter_match], 'c:', lw=0.8)
ax1.plot(iterations, [mean([i[0] for i in p2_iter_match])] * len(iterations), 'c-', lw=1)

# set legends
ax0.legend(loc=6, fontsize=plt_fontsize['legend'])
ax1.legend(loc=7, fontsize=plt_fontsize['legend'])

# set axis label
ax0.set_xlabel('TicTacToe Rounds', fontsize=plt_fontsize['x_lbl'])
ax0.set_ylabel('Cumulative winning rate (%)', fontsize=plt_fontsize['y_lbl'])
ax1.set_ylabel('Iterations', fontsize=plt_fontsize['y_lbl'])

# set tick label
ax0.tick_params(labelsize=plt_fontsize['tick_lbl'])
ax1.tick_params(labelsize=plt_fontsize['tick_lbl'])

# set x, y range
ax0.set_xlim(rrr, mmm*rrr)
ax0.set_ylim(-1, 101)

# set plot title
title_txt = 'MCTS(P1, X) vs MCTS(P2, O)'
if mcts_criteria[0] == 'time':
    title_txt += '\n Time limit: ' + str(mcts_criteria[1]) + 'ms'
else:
    title_txt += '\n Cycles limit: ' + str(mcts_criteria[1]) + '(Cycles)'
ax0.set_title(title_txt, fontsize=plt_fontsize['title'], weight='bold')


# =================================================
plt.tight_layout()
plt.savefig(save_route + '.jpg')
plt.show()