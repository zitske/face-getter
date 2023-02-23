import cv2
import dlib

# Carrega o detector de faces do dlib
detector = dlib.get_frontal_face_detector()

# Inicializa a webcam
cap = cv2.VideoCapture(0)

# Define o tamanho da janela
cv2.namedWindow("Face Detection")

# Contador de fotos
count = 0

while True:
    # Lê a imagem da webcam
    ret, frame = cap.read()

    # Verifica se a imagem é vazia
    if not ret:
        print("Erro: imagem vazia")
        break

    # Converte a imagem para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta as faces na imagem
    faces = detector(gray)

    # Para cada face detectada
    for face in faces:
        # Obtém as coordenadas da face
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        # Recorta a região da face
        face_img = frame[y1:y2, x1:x2]

        # Verifica se a imagem da face não está vazia
        if face_img.size != 0:
            # Salva a imagem da face em um arquivo
            cv2.imwrite(f"face{count}.jpg", face_img)

            # Incrementa o contador de fotos
            count += 1

    # Exibe a imagem na janela
    cv2.imshow("Face Detection", frame)

    # Aguarda a tecla ESC para sair do loop
    key = cv2.waitKey(1)
    if key == 27:
        break

# Libera a webcam
cap.release()

# Fecha a janela
cv2.destroyAllWindows()
