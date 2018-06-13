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

# //PURGED//
# def analysis(matches, rounds, p1_para, p2_para, disp=False):
#     # player parameters : [Type_of_AI, (if MCTS) Type of criteria, Criteria]
#     if p1_para[0] not in ['random, minimax, mcts']: raise NotImplementedError
#     if p2_para[0] not in ['random, minimax, mcts']: raise NotImplementedError
#     if p1_para[0] == 'mcts' and len(p1_para) != 3: raise NotImplementedError
#     if p2_para[0] == 'mcts' and len(p2_para) != 3: raise NotImplementedError
#
#     mmm = matches
#     rrr = rounds
#
#     p1_data, p2_data, tie_data = [], [], []
#     p1_wins, p2_wins, tie = 0, 0, 0
#     iterations = np.arange(rrr, mmm * rrr + 1, rrr)
#
#     if p1_para[0] == 'mcts': p1_iii, p1_iters = np.zeros(rrr), np.zeros(mmm)
#     if p2_para[0] == 'mcts': p2_iii, p2_iters = np.zeros(rrr), np.zeros(mmm)
#
#     time_start = time.time()
#     for i in range(mmm):
#         for j in range(rrr):
#             t = TTT()
#             if disp:
#                 t.dispboard()
#                 print()
#             if p1_para[0] == 'random' and p2_para[0] == 'random':
#                 while t.result == 0:
#                     v = auto_random(t.board)
#                     t.ai_input(v)
#
#                     if disp:
#                         t.dispboard()
#                         print()
#                     t.checkresult()
#                     t.switch_player()
#
#             elif p1_para[0] == 'mcts' and p2_para[0] == 'mcts':
#                 while t.result == 0:
#                     if t.player == 1:
#                         v = auto_mcts(t.board, p1_para[1], p1_para[2])
#                         p1_iii[j] = v[0]
#                         t.ai_input(v[1])
#                     else:
#                         v = auto_mcts(t.board, p2_para[1], p2_para[2])
#                         p2_iii[j] = v[0]
#                         t.ai_input(v[1])
#
#                     if disp:
#                         t.dispboard()
#                         print()
#                     t.checkresult()
#                     t.switch_player()
#
#             elif p1_para[0] == 'random' and p2_para[0] == 'mcts':
#                 while t.result == 0:
#                     if t.player == 1:
#                         v = auto_random(t.board)
#                         t.ai_input(v)
#                     else:
#                         v = auto_mcts(t.board, p2_para[1], p2_para[2])
#                         p2_iii[j] = v[0]
#                         t.ai_input(v[1])
#
#                     if disp:
#                         t.dispboard()
#                         print()
#                     t.checkresult()
#                     t.switch_player()
#
#             elif p1_para[0] == 'mcts' and p2_para[0] == 'random':
#                 while t.result == 0:
#                     if t.player == 1:
#                         v = auto_random(t.board)
#                         t.ai_input(v)
#                     else:
#                         v = auto_mcts(t.board, p2_para[1], p2_para[2])
#                         p2_iii[j] = v[0]
#                         t.ai_input(v[1])
#
#                     if disp:
#                         t.dispboard()
#                         print()
#                     t.checkresult()
#                     t.switch_player()
#
#             r = t.result
#             if disp: print(f'\n{i}/{mmm}, {j}/{rrr}, {r}\n\n')
#             if r == 1: p1_wins += 1
#             elif r == 2: p2_wins += 1
#             else: tie += 1
#
#         p1_data.append(p1_wins / iterations[i] * 100)
#         p2_data.append(p2_wins / iterations[i] * 100)
#         tie_data.append(tie / iterations[i] * 100)
#
#         if p1_para[0] == 'mcts':
#             p1_iters[i] = np.mean(p1_iii)
#             p1_iii = np.zeros(rrr)
#         if p2_para[0] == 'mcts':
#             p2_iters[i] = np.mean(p2_iii)
#             p2_iii = np.zeros(rrr)
#
#     print('Elapsed Time: ', time.time() - time_start)
#     print('P1 winning rate: ', p1_data[-1])
#     print(p2_data[-1])
#print(
#     fig, ax0 = plt.subplots()
#
#     ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = MCTS')
#     ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = MCTS')
#     ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
#     ax0.legend(loc=1)
#     ax0.set_xlabel('TicTacToe Rounds')
#     ax0.set_ylabel('Cumulative winning rate (%)')
#     ax0.set_ylim(-1, 101)
#     if p1_para[0] == 'mcts' or p2_para[0] == 'mcts':
#         ax1 = ax0.twinx()
#         ax1.plot(iterations, p1_iters, 'm--', lw=2, label='P1 MCTS Iterations')
#         ax1.plot(iterations, p2_iters, 'c--', lw=2, label='P2 MCTS Iterations')
#         ax1.legend(loc=2)
#         ax1.set_ylabel('Iterations\n(mean value)')
#     titletxt = f'{p1_para[0].upper()}(P1, X) vs {p2_para[0].upper()}(P2, O)'
#     if p1_para[0] == 'mcts' or p2_para[0] == 'mcts': titletxt += f' - {p1_para[1]} limit: {p1_para[2]}'
#     plt.title('MCTS(P1, X) vs MCTS(P2, O) - time limit: 2ms')


mmm, rrr = 20, 50 # Match, Round
mcts_criteria = ['time', 3] # ['time' or 'iter', millisecond or iteration cycle]
fig_dpi = 200 # size of output img
look_through = False

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

print('Elapsed Time: ', time.time() - time_start)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

plt.figure(dpi=fig_dpi)
plt.grid(True)
plt.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = RANDOM')
plt.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = RANDOM')
plt.plot(iterations, tie_data, 'g--', lw=2, label='Tie')
plt.xlabel('TicTacToe Rounds')
plt.ylabel('Cumulative winning rate (%)')
plt.ylim(-1, 101)
plt.title('Random(P1, X) vs Random(P2, O)')
plt.legend()
plt.savefig('output/TicTacToe_Rand_vs_Rand.jpg')

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

print('Elapsed Time: ', time.time() - time_start)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

fig, ax0 = plt.subplots(dpi=fig_dpi)
ax1 = ax0.twinx()
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = RANDOM')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = MCTS')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, mcts_iterations, 'c--', lw=2, label='MCTS Iterations\n(mean value)')
ax0.legend(loc=6)
ax1.legend(loc=7)
ax0.set_xlabel('TicTacToe Rounds')
ax0.set_ylabel('Cumulative winning rate (%)')
ax1.set_ylabel('Iterations')
ax0.set_ylim(-1, 101)
if mcts_criteria[0] == 'time':
    title_txt = 'Random(P1, X) vs MCTS(P2, O) - Time limit: ' + str(mcts_criteria[1])+'ms'
    save_route = 'output/TicTacToe_Rand_vs_MCTS_Timelimit_'+ str(mcts_criteria[1]) + 'ms.jpg'
else:
    title_txt = 'Random(P1, X) vs MCTS(P2, O) - Iteration limit: ' + str(mcts_criteria[1])+'(Cycles)'
    save_route = 'output/TicTacToe_Rand_vs_MCTS_Iterlimit_'+ str(mcts_criteria[1]) + '.jpg'
plt.title(title_txt)
plt.savefig(save_route)

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

print('Elapsed Time: ', time.time() - time_start)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

fig, ax0 = plt.subplots(dpi=fig_dpi)
ax1 = ax0.twinx()
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = MCTS')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = RANDOM')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, mcts_iterations, 'm--', lw=2, label='MCTS Iterations\n(mean value)')
ax0.legend(loc=6)
ax1.legend(loc=7)
ax0.set_xlabel('TicTacToe Rounds')
ax0.set_ylabel('Cumulative winning rate (%)')
ax1.set_ylabel('Iterations')
ax0.set_ylim(-1, 101)

if mcts_criteria[0] == 'time':
    title_txt = 'MCTS(P1, X) vs Random(P2, O) - Time limit: ' + str(mcts_criteria[1])+'ms'
    save_route = 'output/TicTacToe_MCTS_vs_Rand_Timelimit_'+ str(mcts_criteria[1]) + 'ms.jpg'
else:
    title_txt = 'MCTS(P1, X) vs Random(P2, O) - Iteration limit: ' + str(mcts_criteria[1])+'(Cycles)'
    save_route = 'output/TicTacToe_MCTS_vs_Rand_Iterlimit_'+ str(mcts_criteria[1]) + '.jpg'
plt.title(title_txt)
plt.savefig(save_route)

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
                if t.player == 1:
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

print('Elapsed Time: ', time.time() - time_start)
print('P1 winning rate: ', p1_data[-1])
print('P2 winning rate: ', p2_data[-1])
print()

fig, ax0 = plt.subplots(dpi=fig_dpi)
ax1 = ax0.twinx()
ax0.grid(True)
ax0.plot(iterations, p1_data, 'r', lw=3, label='P1(X) = MCTS')
ax0.plot(iterations, p2_data, 'b', lw=3, label='P2(O) = MCTS')
ax0.plot(iterations, tie_data, 'g', lw=2, label='Tie')
ax1.plot(iterations, p1_iterations, 'm--', lw=2, label='P1 MCTS Iterations')
ax1.plot(iterations, p2_iterations, 'c--', lw=2, label='P2 MCTS Iterations')
ax0.legend(loc=1)
ax1.legend(loc=2)
ax0.set_xlabel('TicTacToe Rounds')
ax0.set_ylabel('Cumulative winning rate (%)')
ax1.set_ylabel('Iterations\n(mean value)')
ax0.set_ylim(-1, 101)
if mcts_criteria[0] == 'time':
    title_txt = 'MCTS(P1, X) vs MCTS(P2, O) - Time limit: ' + str(mcts_criteria[1])+'ms'
    save_route = 'output/TicTacToe_MCTS_vs_MCTS_Timelimit_'+ str(mcts_criteria[1]) + 'ms.jpg'
else:
    title_txt = 'MCTS(P1, X) vs MCTS(P2, O) - Iteration limit: ' + str(mcts_criteria[1])+'(Cycles)'
    save_route = 'output/TicTacToe_MCTS_vs_MCTS_Iterlimit_'+ str(mcts_criteria[1]) + '.jpg'
plt.title(title_txt)
plt.savefig(save_route)