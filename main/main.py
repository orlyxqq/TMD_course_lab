import numpy as np 
import math


def create_transition_matrix(buffer_size, lambda_rate):
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
        
    

def sim_metrics():
    pass

def teor_metrics(): 
    pass

def plot_graphics():
    pass

def main():
    #(b + 1)^M < 20
    M = 2 #абоненты  
    b = 4 #буфер
    
    lambda_rate = np.linspace(0.1, 1.0 ,1)
    T = 10000 #таймслоты для симуляци
    p =[0.1, 0.2] #вероятность передачи сообщения абонентами
    transition_matrix = create_transition_matrix(b, 0.2 )
    
    print(transition_matrix)
    
    
    
if __name__ == "__main__":
    main()
    

