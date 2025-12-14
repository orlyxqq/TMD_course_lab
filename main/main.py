'''
Смоделировать систему: 
1) В которой 𝑀 абонентов, 𝑀 < ∞ 
2) Каждый абонент имеет буфер одинаковой длинны, 𝑏 < ∞ 
3) Каждый абонент передает в канал сообщение со своей вероятностью (𝑝1,𝑝2,…,𝑝𝑀) 
4) На вход каждого абонента поступает Пуассоновский входной поток сообщений с 
интенсивностью (𝜆1,𝜆2,…,𝜆𝑀) 
Описать работу системы с помощью многомерной Марковской цепи зафиксировав 𝑀 и 𝑏 
таким образом, чтобы (𝑏 + 1)^𝑀 < 20. Каждое состояние Марковской цепи — это вектор 
длинны 𝑀, компоненты вектора это количеством сообщений в буфере у абонентов. 
Значения 𝑀 и 𝑏 согласовать с преподавателем. Сформировать матрицу переходных 
вероятностей вручную для конкретных 𝑀 и 𝑏. 

Путем моделирования и расчета при фиксированном значении 𝑀 и 𝑏 пполучить 
зависимости от 𝜆 следующих характеристик: 
1) 𝑁  – среднее число сообщений
2) 𝑑  – средняя задержка 
'''


import numpy as np 
from collections import deque
import itertools
import tqdm
import numpy as np
from itertools import product
import math
import matplotlib.pyplot as plt

def poisson_prob(k, lam):
    return math.exp(-lam) * lam**k / math.factorial(k)

def create_transition_matrix(arrival_rates, transmit_probs, buffer_size):
    
    num_users = len(arrival_rates)
    num_states = (buffer_size + 1) ** num_users

    all_states = list(product(range(buffer_size + 1), repeat=num_users))
    state_index = {s: i for i, s in enumerate(all_states)}

    P = np.zeros((num_states, num_states))

    for idx, state in enumerate(all_states):

        
        transmit_options = []
        for i in range(num_users):
            if state[i] > 0:
                transmit_options.append([(0, 1 - transmit_probs[i]), (1, transmit_probs[i])])
            else:
                transmit_options.append([(0, 1.0)]) 

        for transmit_combo in product(*transmit_options):
            
            contenders = [i for i, t in enumerate(transmit_combo) if t[0] == 1]
            
            p_tx = np.prod([t[1] for t in transmit_combo])

            if len(contenders) <= 1:
                transmitted = [0] * num_users
                if len(contenders) == 1:
                    transmitted[contenders[0]] = 1
            else:
                transmitted = [0] * num_users

            state_after_tx = tuple(max(state[i] - transmitted[i], 0) for i in range(num_users))

            arrivals_list = []
            for i, lam in enumerate(arrival_rates):
                max_arrival = buffer_size - state_after_tx[i]
                arrivals_prob = [poisson_prob(k, lam) for k in range(max_arrival + 1)]
                arrivals_prob[-1] = 1 - sum(arrivals_prob[:-1])
                arrivals_list.append(list(enumerate(arrivals_prob)))

            for arrivals_combo in product(*arrivals_list):
                next_state = tuple(min(state_after_tx[i] + arrivals_combo[i][0], buffer_size)
                                   for i in range(num_users))
                p_arr = np.prod([arrivals_combo[i][1] for i in range(num_users)])
                P[idx, state_index[next_state]] += p_tx * p_arr

    return P

def sim_metrics(arrival_rates, transmit_probs, buffer_size, time_windows):

    num_users = len(arrival_rates)

    queues = [deque(maxlen=buffer_size) for _ in range(num_users)]
    processed_requests = [0] * num_users
    total_delay = [0.0] * num_users
    avg_queue_over_time = [[] for _ in range(num_users)]

    for current_window in tqdm.tqdm(range(time_windows), ncols=80, desc="magic"):

        window_start = current_window
        window_end = window_start + 1

        contenders = []
        for u in range(num_users):
            if queues[u] and np.random.rand() < transmit_probs[u]:
                contenders.append(u)

        if len(contenders) == 1:
            u = contenders[0]
            arrival_time = queues[u].popleft()
            delay = (window_end - arrival_time) 
            total_delay[u] += delay
            processed_requests[u] += 1

        for u in range(num_users):
            n_arr = np.random.poisson(arrival_rates[u])
            if n_arr > 0:
                arr_times = np.random.uniform(window_start, window_end, n_arr)
                arr_times.sort()
                for t in arr_times:
                    if len(queues[u]) < buffer_size:
                        queues[u].append(t)

        for u in range(num_users):
            avg_queue_over_time[u].append(len(queues[u]))

    avg_delay = [
        total_delay[u] / processed_requests[u] if processed_requests[u] > 0 else 0.0
        for u in range(num_users)
    ]
    avg_queue = [
        sum(avg_queue_over_time[u]) / len(avg_queue_over_time[u])
        for u in range(num_users)
    ]
    lambd_out = [processed_requests[u] / time_windows for u in range(num_users)]

    return {
        'svd': avg_delay,
        'svr': avg_queue,
        'lambdout': lambd_out
    }
def teor_metrics(transition_matrix, arrival_rates, transmit_probs, buffer_size):

    P = np.asarray(transition_matrix)
    n = P.shape[0]

    A = P.T - np.eye(n)
    A[-1, :] = 1.0
    b = np.zeros(n)
    b[-1] = 1.0
    pi = np.linalg.solve(A, b)

    num_users = len(transmit_probs)
    all_states = list(product(range(buffer_size + 1), repeat=num_users))

    # 1. Среднее число сообщений в буфере
    avg_queue = []
    for u in range(num_users):
        q_u = sum(pi[i] * all_states[i][u] for i in range(n))
        avg_queue.append(q_u)

    # 2. Выходная интенсивность λ_out (throughput)
    lambd_out = []
    for u in range(num_users):
        th_u = 0.0
        p_u = transmit_probs[u]
        for i, state in enumerate(all_states):

            if state[u] == 0:
                continue

            prob_others_not_tx = 1.0
            for j in range(num_users):
                if j == u:
                    continue
                if state[j] > 0:
                    prob_others_not_tx *= (1 - transmit_probs[j])

            success_prob = p_u * prob_others_not_tx
            th_u += pi[i] * success_prob


        lambd_out.append(th_u)

    # 3. Средняя задержка (Little’s law + 0.5)
    avg_delay = []
    for u in range(num_users):
        if lambd_out[u] > 0:
            avg_delay.append(avg_queue[u] / lambd_out[u] + 0.5)
        else:
            avg_delay.append(0.0)

    return {
        'tvd': avg_delay,          # задержка
        'tvr': avg_queue,          # среднее число сообщений
        'lambdout': lambd_out     
    }


def main():
    #(b + 1)^M < 20
    
    M = 3 #абоненты  
    b = 2 #буфер
    
    # lambda_rate_list = [0.7, 0.7, 0.7] #входная интенсивность
    p_transmit_list = [1/2, 1/5, 1/5] #вероятность передачи сообщения абонентами
    # time_windows = 100000 #таймслоты
    
    
    # sim_metrics_abonents = sim_metrics(lambda_rate_list, p_transmit_list, b, time_windows)
    # transition_matrix = create_transition_matrix(lambda_rate_list ,p_transmit_list, b)
    # teor_metrics_abonents = teor_metrics(transition_matrix, lambda_rate_list, p_transmit_list, b)
    
    # print(transition_matrix)
    # print(sim_metrics_abonents)
    # print(teor_metrics_abonents)
    
    lambda_values = np.linspace(0.1, 1.0, 10)

    sim_delay = [[] for _ in range(M)]
    teor_delay = [[] for _ in range(M)]
    sim_queue = [[] for _ in range(M)]
    teor_queue = [[] for _ in range(M)]
    sim_lambdas = [[] for _ in range(M)]
    teor_lambdas = [[] for _ in range(M)]

    time_windows = 20000 

    for lam in lambda_values:
        lambda_rate_list = [lam] * M
        lambda_rate_list[1] = 0.05
        lambda_rate_list[2] = 0.05
        print(lambda_rate_list)
        

        sim_res = sim_metrics(lambda_rate_list, p_transmit_list, b, time_windows)

        P = create_transition_matrix(lambda_rate_list, p_transmit_list, b)
        teor_res = teor_metrics(P, lambda_rate_list, p_transmit_list, b)

        for u in range(M):
            sim_delay[u].append(sim_res['svd'][u])
            teor_delay[u].append(teor_res['tvd'][u])
            sim_queue[u].append(sim_res['svr'][u])
            teor_queue[u].append(teor_res['tvr'][u])
            sim_lambdas[u].append(sim_res['lambdout'][u])
            teor_lambdas[u].append(teor_res['lambdout'][u])
    



    plt.figure(figsize=(7,5))
    for u in range(M):
        plt.plot(lambda_values, teor_delay[u], label=f"Теория, абонент {u+1}")
        plt.plot(lambda_values, sim_delay[u], '--', label=f"Симуляция, абонент {u+1}")
    plt.title("Средняя задержка")
    plt.xlabel("λ")
    plt.ylabel("средняя задержка")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure(figsize=(7,5))
    for u in range(M):
        plt.plot(lambda_values, teor_queue[u], label=f"Теория, абонент {u+1}")
        plt.plot(lambda_values, sim_queue[u], '--', label=f"Симуляция, абонент {u+1}")
    plt.title("Среднее число сообщений")
    plt.xlabel("λ")
    plt.ylabel("Среднее число сообщений")
    plt.grid(True)
    plt.legend()
    plt.show()
    
    plt.figure(figsize=(7,5))
    for u in range(M):
        plt.plot(lambda_values, teor_lambdas[u], label=f"Теория λ_out абонент {u+1}")
        plt.plot(lambda_values, sim_lambdas[u], '--', label=f"Симуляция λ_out абонент {u+1}")
    plt.title("Выходная интенсивность λ_out")
    plt.xlabel("λ_in")
    plt.ylabel("λ_out")
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
