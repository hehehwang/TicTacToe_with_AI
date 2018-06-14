import random

def random_action(s):
    actions = available_action(s)
    return random.choice(actions)

def available_action(s):
    l = []
    for i in range(9):
        if s[i] == '_': l.append(i)
    return l