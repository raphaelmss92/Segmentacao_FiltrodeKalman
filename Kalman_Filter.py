import numpy as np

# http://ros-developer.com/2019/04/10/kalman-filter-explained-with-python-code-from-scratch/

class kalman_filter(object):
    '''
    Filtro de Kalman: Observador de estados ótimo, com capacidade de balancear medidas ruidosas e erros de modelagem.
        As matrizes Q e R controlam grau de confiança no modelo em espaço de estados ou na medição. 
   
    Matrizes no modelo de espaço de estados:
    A: matriz de transição de estados 
    B: matriz de controle              
    C: matriz de saída                
    D: matriz de ligação direta       

    initial_P: matriz de covariância inicial 
    Q: variância do modelo
    R: variância da medição
    initial_X: matriz de estados inicial - n_estados x 1

    '''
    def __init__(self, A, B, C, D, initial_P, Q, R, initial_X):
        
        self.A = A # n_estados x n_estados
        self.B = B # n_estados x n_estradas
        self.C = C # n_estados_desejados x n_estados (1's nos estados desejados - Que serão fornecidos pela medição)
        self.D = D # ---
        self.Q = Q # n_estados x n_estados (matriz diagonal)
        self.R = R # n_estados_des x n_estados_des (matriz diagonal)

        self.last_X = initial_X # n_estados x 1
        self.X_predicted = None
        self.X_ajusted = None

        self.last_P = initial_P # n_estados x n_estados (matriz diagonal)
        self.P_predicted = None
        self.P_ajusted = None

        self.K = 0 # Ganho do observador

        # Predição:
        # x = Ax + Bu
        # (n_estados, 1) = (n_estados, n_estados)x(n_estados, 1) + (n_estados, n_entradas)x(n_entradas, 1)

        # P = APA^T + Q
        # Todos são n_estados x n_estados

        # Atualização:
        # K = PC^T(CPC^T + R)^-1
        # (n_estados, n_estados_des) = (n_estados, n_estados)x(n_estados, n_estados_des)x((n_estados_des, n_estados)x(n_estados, n_estados)x(n_estados, n_estados_des) + (n_estados_des, n_estados_des))

        # x = x_p + k(Y - Cx_p) -> Equação do observador de estados
        #(n_estados, 1) = (n_estados, 1) + (n_estados, n_estados_des)x((n_estados_des, 1) - (n_estados_des, n_estados)x(n_estados, 1))

        # P = (I - KC)P
        # (n_estados, n_estados) = ((n_estados, n_estados) - (n_estados, n_estados_des)x(n_estados_des, n_estados))(n_estados, n_estados)

    def predict(self, U):
        # U - (n_entradas, 1)

        self.X_predicted = self.A @ self.last_X + self.B @ U
        self.P_predicted = self.A @ (self.last_P @ self.A.T) + self.Q


    def update(self, Y_measured):
        # Y - (n_estados_desejados, 1)

        num_K = self.P_predicted @ self.C.T
        den_K = self.C @ num_K + self.R

        self.K = num_K @ np.linalg.inv(den_K)    

        self.X_ajusted = self.X_predicted + self.K @ (Y_measured - self.C @ self.X_predicted)
        self.P_ajusted = ((np.eye(self.P_predicted.shape[0]) - self.K @ self.C) @ (self.P_predicted))*np.eye(self.P_predicted.shape[0])

        self.last_P = self.P_ajusted
        self.last_X = self.X_ajusted
        