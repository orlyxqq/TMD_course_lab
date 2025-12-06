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


def create_transition_matrix(buffer_size, p, lambda_rate):
    size = (buffer_size + 1)
    transition_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            if i == 0:
                if j < buffer_size:
                    transition_matrix[i][j] = (lambda_rate ** j) * math.exp(-lambda_rate) / math.factorial(j)
                else:
                    transition_matrix[i][j] = 1 - sum(transition_matrix[i][:buffer_size])
            elif i == buffer_size:
                if j == buffer_size - 1:
                    transition_matrix[i][j] = 1.0
                else:
                    transition_matrix[i][j] = 0.0
            else:
                if j == size - 1:
                    transition_matrix[i][j] = 0
                elif j >= i - 1 and j == buffer_size - 1: #последний переход
                    transition_matrix[i][j] = 1 - sum(transition_matrix[i][:buffer_size])
                elif j >= i - 1 and j < buffer_size:
                    k = j - (i - 1)
                    transition_matrix[i][j] = (lambda_rate ** k) * math.exp(-lambda_rate) / math.factorial(k)
                else:
                    transition_matrix[i][j] = 0    
                
    return transition_matrix
        
    

def sim_metrics(arrival_rates, transmit_probs, buffer_size, time_windows):
    """
    arrival_rates   : list of floats (λ_i) — интенсивность генерации каждого абонента
    transmit_probs  : list of floats (p_i) — вероятность передачи в каждом слоте
    buffer_size     : int    — размер буфера
    time_windows    : int    — количество временных окон

    Возвращает:
        svd[i] — средняя задержка по абоненту i
        svr[i] — среднее число заявок в очереди
        dropped[i] — потерянные пакеты
    """

    num_users = len(arrival_rates)

    # Очереди
    queues = [deque(maxlen=buffer_size) for _ in range(num_users)]

    # Метрики
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
    b = 4 #буфер
    
    lambda_rate_i = [0.3, 0.2]
    T = 10000 #таймслоты для симуляци
    P_i = [0.1, 0.9] #вероятность передачи сообщения абонентами
    
    
    transition_matrix = create_transition_matrix(b, P_i[0], 0.2 )
    teor_metrics_abonents = sim_metrics(lambda_rate_i, P_i, b, T)
    
    print(teor_metrics_abonents)
    
    
    
if __name__ == "__main__":
    main()
    

