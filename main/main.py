'''
Смоделировать систему: 
1) В которой 𝑀 абонентов, 𝑀 < ∞ 
2) Каждый абонент имеет буфер одинаковой длинны, 𝑏 < ∞ 
3) Каждый абонент передает в канал сообщение со своей вероятностью (𝑝1,𝑝2,…,𝑝𝑀) 
4) На вход каждого абонента поступает Пуассоновский входной поток сообщений с 
интенсивностью (𝜆1,𝜆2,…,𝜆𝑀) 
Описать работу системы с помощью многомерной Марковской цепи зафиксировав 𝑀 и 𝑏 
таким образом, чтобы (𝑏 + 1)𝑀 < 20. Каждое состояние Марковской цепи — это вектор 
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


def create_transition_matrix(M, B, lambda_rate, p):
    """
    Универсальная матрица переходов для M абонентов и буфера B.
    y[i] = exp(-lambda[i])
    p[i] – вероятность успеха передачи.
    """

    # --- вероятность появления сообщения ---
    y = np.exp(-np.array(lambda_rate))

    # --- все состояния ---
    states = list(itertools.product(range(B + 1), repeat=M))
    S = len(states)
    index = {s: i for i, s in enumerate(states)}

    P = np.zeros((S, S))

    for s_idx, state in enumerate(states):

        for T in range(M):                     # кто передаёт
            for success in [0, 1]:             # успех передачи
                for arrivals in itertools.product([0, 1], repeat=M):

                    # --- вероятность события ---
                    prob = 1.0

                    # выбор передающего
                    prob *= 1.0 / M

                    # успех/неудача передачи
                    prob *= p[T] if success else (1 - p[T])

                    # приходы сообщений
                    for i in range(M):
                        prob *= y[i] if arrivals[i] else (1 - y[i])

                    if prob == 0:
                        continue

                    # --- формируем следующее состояние ---
                    new_state = list(state)

                    for i in range(M):

                        # уход сообщения
                        if i == T and state[i] > 0 and success:
                            new_state[i] -= 1

                        # приход сообщения
                        if arrivals[i] == 1 and new_state[i] < B:
                            new_state[i] += 1

                    new_state = tuple(new_state)
                    new_idx = index[new_state]

                    P[s_idx, new_idx] += prob

    return P
    

def sim_metrics(arrival_rates, transmit_probs, buffer_size, time_windows):

    num_users = len(arrival_rates)

    # Очереди
    queues = [deque(maxlen=buffer_size) for _ in range(num_users)]

    processed = [0] * num_users
    dropped = [0] * num_users
    total_delay = [0.0] * num_users
    avg_queue_over_time = [[] for _ in range(num_users)]

    for current_window in tqdm.tqdm(range(time_windows), ncols=80, desc="Simulation"):

        window_start = current_window
        window_end = current_window + 1

        for i in range(num_users):
            n_arr = np.random.poisson(arrival_rates[i])
            if n_arr > 0:
                arr_times = np.random.uniform(window_start, window_end, n_arr)
                arr_times.sort()

                for t in arr_times:
                    if len(queues[i]) < buffer_size:
                        queues[i].append(t)
                    else:
                        dropped[i] += 1

            avg_queue_over_time[i].append(len(queues[i]))

        contenders = []   # абоненты, которые пытаются передать

        for i in range(num_users):
            if queues[i] and np.random.rand() < transmit_probs[i]:
                contenders.append(i)

        if len(contenders) == 1:
            user = contenders[0]
            arrival_time = queues[user].popleft()

            delay = (window_end - arrival_time) + 1
            total_delay[user] += delay
            processed[user] += 1

    avg_delay = [
        total_delay[i] / processed[i] if processed[i] > 0 else 0.0
        for i in range(num_users)
    ]

    avg_queue = [
        sum(avg_queue_over_time[i]) / len(avg_queue_over_time[i])
        for i in range(num_users)
    ]

    return {
        'svd': avg_delay,
        'svr': avg_queue,
        'dropped': dropped
    }

def teor_metrics(): 
    pass

def plot_graphics():
    pass

def main():
    #(b + 1)^M < 20
    
    M = 2 #абоненты  
    b = 1 #буфер
    
    lambda_rate_list = [1.0, 1.0]
    p_transmit_list = [1.0, 1.0] #вероятность передачи сообщения абонентами
    time_windows = 10000 #таймслоты для симуляци
    
    
    teor_metrics_abonents = sim_metrics(lambda_rate_list, p_transmit_list, b, time_windows)
    transition_matrix = create_transition_matrix(M, b, lambda_rate_list ,p_transmit_list)
    print(transition_matrix)
    #print(teor_metrics_abonents)
    
    
    
if __name__ == "__main__":
    main()
    
    
 
  
    

