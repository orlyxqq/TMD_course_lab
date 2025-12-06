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
import math
from collections import deque
import tqdm
from scipy.stats import poisson
from itertools import product


def state_to_index(state, b):
    """Перевод состояния (n1,...,nM) в индекс"""
    M = len(state)
    index = 0
    for i, n in enumerate(state):
        index += n * (b+1)**(M-1-i)
    return index

def index_to_state(index, M, b):
    """Перевод индекса в состояние"""
    state = []
    for i in range(M):
        div = (b+1)**(M-1-i)
        n = index // div
        state.append(n)
        index %= div
    return tuple(state)

def create_transition_matrix(lambda_rates, p_transmit_list, b):
    """
    Создание матрицы переходов для M абонентов
    lambda_rates: list[float] λ_i
    p_transmit_list: list[float] p_i
    b: int, размер буфера каждого абонента
    """
    M = len(lambda_rates)
    size = (b+1)**M
    P = np.zeros((size, size))

    # Перебираем все состояния системы
    for state in product(range(b+1), repeat=M):
        idx_from = state_to_index(state, b)

        # --- Генерация всех возможных комбинаций прибывших пакетов ---
        # Берем до b пакетов, можно оптимизировать ограничением
        arrival_ranges = [range(b+1) for _ in range(M)]

        for arrivals in product(*arrival_ranges):
            prob_arrival = 1.0
            next_state = list(state)

            for i in range(M):
                # Poisson вероятность k пакетов
                k = arrivals[i]
                if state[i] + k > b:
                    k = b - state[i]  # обрезка буфера
                prob_arrival *= poisson.pmf(k, lambda_rates[i])
                next_state[i] = state[i] + k

            # --- Выбор абонента для передачи пакета ---
            contenders = [i for i in range(M) if state[i] > 0]

            success_probs = []

            # Генерируем все комбинации попыток передачи
            # Каждая комбинация — бинарный вектор длины M
            # prob_attempt[i] = p_i если пакет есть, иначе 0
            attempts_probs = []
            for i in range(M):
                if state[i] > 0:
                    attempts_probs.append([1 - p_transmit_list[i], p_transmit_list[i]])
                else:
                    attempts_probs.append([1.0, 0.0])

            for attempt in product(*attempts_probs):
                # Кто пытается передать
                num_attempts = sum(attempt)
                next_state_final = next_state.copy()
                prob_attempt_comb = np.prod([ap for ap in attempt])
                if num_attempts == 1:
                    # успешная передача → уменьшаем соответствующий буфер
                    idx_success = attempt.index(1)
                    next_state_final[idx_success] = max(0, next_state_final[idx_success] - 1)
                # иначе коллизия → никто не передает

                idx_to = state_to_index(next_state_final, b)
                P[idx_from, idx_to] += prob_arrival * prob_attempt_comb

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
    
    lambda_rate_i = [0.3, 0.2]
    P_i = [0.1, 0.9] #вероятность передачи сообщения абонентами
    
    T = 10000 #таймслоты для симуляци
    
    
    
    teor_metrics_abonents = sim_metrics(lambda_rate_i, P_i, b, T)
    teor_matrix = create_transition_matrix(lambda_rate_i, P_i, b)
    print(teor_matrix)
    #print(teor_metrics_abonents)
    
    
    
if __name__ == "__main__":
    main()
    
    
 
  
    

