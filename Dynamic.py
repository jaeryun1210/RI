# -*- coding: utf-8 -*-
import numpy as np

# 상태 수 및 설정
num_states = 10
gamma = 0.9  # 할인율
theta = 1e-4  # 수렴 기준

# 전이 정보 정의 (확률, 다음 상태, 보상)
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
    9: []  # 종료
}

# 정책 이터레이션
def policy_iteration(transitions, gamma=0.9, theta=1e-4):
    V = np.zeros(num_states)
    policy = {s: None for s in range(num_states)}
    
    is_policy_stable = False
    while not is_policy_stable:
        # 정책 평가
        while True:
            delta = 0
            for s in range(num_states):
                v = V[s]
                V[s] = sum(p * (r + gamma * V[s_]) for (p, s_, r) in transitions.get(s, []))
                delta = max(delta, abs(v - V[s]))
            if delta < theta:
                break

        # 정책 개선
        is_policy_stable = True
        for s in range(num_states):
            actions = transitions.get(s, [])
            if not actions:
                continue
            
            # 탐욕 정책
            best_action = max(actions, key=lambda x: x[0] * (x[2] + gamma * V[x[1]]))
            
            # 이전 정책과 비교하여 변경 여부 확인
            if policy[s] != best_action:
                is_policy_stable = False
            policy[s] = best_action
    
    return V, policy

V_result, policy_result = policy_iteration(transitions, gamma, theta)

# 상태별 가치 함수 출력
print("State   Value")
for s in range(num_states):
    print(f"   s{s}  {V_result[s]:.4f}")
# 최적 정책 출력
print("\nState  Best Action  Next State  Reward")
for s, action in policy_result.items():
    if action:
        print(f"   s{s}      {action[1]:<10} {action[1]:<10} {action[2]:<10}")
    else:
        print(f"   s{s}      TERMINAL")
