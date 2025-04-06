import numpy as np
import pandas as pd

num_states = 10
gamma = 0.9
num_episodes = 10000

V = np.zeros(num_states)
Returns = {s: [] for s in range(num_states)}

transitions = {
    0: [(0.5, 1, 0), (0.5, 2, 0)],
    1: [(0.4, 0, 0), (0.3, 2, 1), (0.3, 8, -1)],
    2: [(0.8, 3, 0), (0.2, 5, 0)],
    3: [(0.6, 0, -1), (0.4, 4, 0)],
    4: [(0.5, 5, 1), (0.5, 2, 0)],
    5: [(0.4, 2, 1), (0.3, 1, -1), (0.3, 8, -2)],
    6: [(0.2, 4, 0), (0.4, 7, 0), (0.4, 9, 10)],
    7: [(0.2, 6, 0), (0.8, 9, 2)],
    8: [(0.6, 7, 0), (0.4, 8, 0)],
    9: []
}

# 에피소드 생성 함수
def generate_episode():
    episode = []
    state = 0
    while state != 9:
        actions = transitions[state]
        if not actions:
            break
        probs = [p for (p, _, _) in actions]
        next_states = [s_ for (_, s_, _) in actions]
        rewards = [r for (_, _, r) in actions]
        action_idx = np.random.choice(len(actions), p=probs)
        next_state = next_states[action_idx]
        reward = rewards[action_idx]
        episode.append((state, reward))
        state = next_state
    episode.append((9, 0))
    return episode

for episode_idx in range(num_episodes):
    episode = generate_episode()
    G = 0
    visited = set()
    for t in reversed(range(len(episode))): #에피소드 뒤에서부터 리턴계산
        state, reward = episode[t]
        G = gamma * G + reward    #감가율 적용후 누적 리턴계산
        if state not in visited:
            Returns[state].append(G)
            V[state] = np.mean(Returns[state])
            visited.add(state)

state_list = [f"s{i}" for i in range(num_states)]
value_list = V.round(4)
result = pd.DataFrame({
    "State": state_list,
    "Value": value_list
})
print(result.to_string(index=False))
