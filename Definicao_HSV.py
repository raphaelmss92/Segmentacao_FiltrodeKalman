import cv2
import numpy as np

def nothing(x):
    pass

camPort = 'http://192.168.0.107:4747/video'
camera = cv2.VideoCapture(camPort)

# Criando barras deslizantes para controle dos intervalos de segmentação das cores
cv2.namedWindow('Trackbars')

cv2.createTrackbar('H - Lower', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('S - Lower', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('V - Lower', 'Trackbars', 0, 255, nothing)
cv2.createTrackbar('H - Upper', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S - Upper', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V - Upper', 'Trackbars', 255, 255, nothing)


while(True):

    # Obtendo frame da câmera
    ret, frame = camera.read()

    # Aplicando filtro passa baixa gaussiano
    frame = cv2.GaussianBlur(frame, (5, 5), 0)

    # Transformando o frame no formato HSV
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Obtendo valores para aplicação de máscara sobre a imagem
    h_l = cv2.getTrackbarPos('H - Lower', 'Trackbars')
    s_l = cv2.getTrackbarPos('S - Lower', 'Trackbars')
    v_l = cv2.getTrackbarPos('V - Lower', 'Trackbars')
    h_u = cv2.getTrackbarPos('H - Upper', 'Trackbars')
    s_u = cv2.getTrackbarPos('S - Upper', 'Trackbars')
    v_u = cv2.getTrackbarPos('V - Upper', 'Trackbars')

    # Valor mínimo
    lower_Color = np.array([h_l, s_l, v_l])

    # Valor máximo
    upper_Color = np.array([h_u, s_u, v_u])

    # Criando a máscara
    mask = cv2.inRange(frameHSV, lower_Color, upper_Color)

    # Aplicando a máscara
    imageMask = cv2.bitwise_and(frame, frame, mask=mask)

    # Função de detecção de bordas
    contornos = cv2.Canny(imageMask, 100, 150)

    # Função que encontra contornos na imagem
    cnts, _ = cv2.findContours(contornos.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Exibindo imagem com a máscara aplicada e os contornos
    cv2.imshow('Frame', imageMask)
    cv2.imshow('Contornos', contornos)

    # Apertar ESC para sair
    k = cv2.waitKey(25)
    if k==27:
        camera.release()
        cv2.destroyAllWindows()
        break

# Printando os intervalos definidos:
print(lower_Color)
print(upper_Color)



