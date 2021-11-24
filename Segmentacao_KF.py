import time
import numpy as np
import cv2
from Kalman_Filter import kalman_filter

# Definições iniciais ==================================================================================================

# Definindo câmera. Possui IP:porta pois foi conexão ao celular através da rede, pelo app DroidCam.
cam  = cv2.VideoCapture('http://192.168.0.107:4747/video')

# Tempo para predição de posição e velocidade do objeto no modelo em espaço de estados. Iniciado aleatóriamente.
T = 0.05

# Modelo em espaço de estados de um sistema de particula no espaço bidimensional
U = np.array([[0], [0]]) # A entrada nesse caso (aceleração em x e y) é sempre nula
A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
B = np.array([[0.5*T**2, 0], [0, 0.5*T**2], [T, 0], [0, T]])
C = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
D = 0

# Estados: [x; y; v_x; v_y] -> posições x, y e velocidades v_x, v_y

# Definindo matrizes do filtro
Q = np.eye(4)*0.01
R = np.diag([0.01, 0.01])

initial_P = np.eye(4)*1e-2
initial_X = np.array([[0],[0],[0],[0]])

# Criando o filtro de kalman
KF = kalman_filter(A, B, C, D, initial_P, Q, R, initial_X)

# Valor mínimo para segmentação 
lower_color = np.array([0, 165, 120])

# Valor máximo para segmentação
upper_color = np.array([179, 226, 190])

def get_measure(img):
    '''
    Obtém a posição do objeto segmentado
    '''
    sum_x = img.sum(axis=0)
    sum_y = img.sum(axis=1)
    
    x = np.argmax(sum_x)
    y = np.argmax(sum_y)
    
    return x, y


# Laço de repetição ====================================================================================================

while True:

    cnf, img = cam.read()

    if cnf:
        
        strt = time.time()

        # Aplicando filtro gaussiano contra ruídos na imagem
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Transformando imagem no formato HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Criando a máscara para segmentação
        mask = cv2.inRange(img_hsv, lower_color, upper_color)

        # Aplicando a máscara
        img_mask = cv2.bitwise_and(img, img, mask=mask)

        # Lendo a posição do objeto segmentado
        x, y = get_measure(mask)
        
        # Atualização e predição do filtro de Kalman
        KF.predict(U)
        KF.update(np.array([[x],[y]]))
        
        x_kf, y_kf, _, _ = KF.X_predicted.reshape(-1)
        
        x_kf = np.int(np.round(np.clip(x_kf, 0, 640)))
        y_kf = np.int(np.round(np.clip(y_kf, 0, 480)))
        
        # Desenhando na imagem as posições medida e predita
        cv2.circle(img_mask, (x, y), 10, (255, 255, 255), -1)
        cv2.circle(img_mask, (x_kf, y_kf), 5, (0, 255, 0), -1)
        
        # Textos para identificação da posição medida e da obtida pelo filtro de kalman
        cv2.putText(img_mask, "Medida", (0,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)            
        cv2.putText(img_mask, "Kalman", (0,30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
        
        # Visualizando resultados
        cv2.imshow('image', img_mask)

        ipt = cv2.waitKey(25)
        
        end = time.time()

        # Se pressionar tecla ESC, sair do loop e finalizar
        if ipt == 27:
            cam.release()
            cv2.destroyAllWindows()
            break
        
        T = strt - end
            
        
