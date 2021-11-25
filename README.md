# Segmentacao_FiltrodeKalman
Uso de segmentação de imagens com filtro de kalman para rastrear a posição de um objeto em vídeo.

Neste projeto, a *segmentação de imagem* é feita através da transformação da imagem obtida pela câmera em HSV, que transforma as camadas BGR (openCV trabalha com BGR ao invés de RGB) em tonalidade, saturação e brilho, permitindo um controle melhor na criação de máscaras nos intervalos de valores desejados. A medida de posição do objeto é obtida pela soma dos valores acumulados em cada eixo da máscara, encontrando a posição de maior valor acumulado.

O *Filtro de Kalman* é um observador de estados ótimo que é capaz de lidar com medições ruidosas e modelos com erros, criando um balanceamento destes através de um escalonamento dos ganhos do observador de modo interativo. Nesta aplicação, o modelo em espaço de estados utilizado no filtro é relacionado à dinâmica de uma partícula no plano bidimensional (x, y), já que a visualização da imagens é neste plano. Assim, o filtro recebe a medição obtida pela segmentação e através do modelo realiza uma predição baseada em ambos, retornando uma medição mais "comportada". É possível controlar o quanto de confiança aplicar no modelo ou na medição. Se focar muito na medição, e saída do filtro irá acompanhar quase perfeitamente a medida, podendo replicar o ruído. Caso foque mais no modelo, este pode gerar predições mais lentas e não acompanhar variações rápidas na medição.

O GIF abaixo demonstra o projeto em funcionamento:

![image 2021-11-24 10-20-58](https://user-images.githubusercontent.com/88464241/143253781-9ba231a7-6fb0-45fb-88a3-01387664091a.gif)
