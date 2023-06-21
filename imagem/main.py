import cv2
import mediapipe as mp

# recohecer a webcam do computador
webcam = cv2.VideoCapture(0)

# reconhecer as maos da pessoa
reconhecer_maos = mp.solutions.hands
maos = reconhecer_maos.Hands()
desenho_mp = mp.solutions.drawing_utils

if webcam.isOpened():
    validacao, frame = webcam.read()
    # pegar todos os frames da tela
    while validacao:
        validacao, frame = webcam.read()

        # converter bgr para rgb
        frame_rbg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        lista_maos = maos.process(frame_rbg)

        # desenhar as mãos
        if lista_maos.multi_hand_landmarks:
            for mao in lista_maos.multi_hand_landmarks:
                desenho_mp.draw_landmarks(frame, mao, reconhecer_maos.HAND_CONNECTIONS)

        # mostrar cada frame que o python esta vendo
        cv2.imshow('webcam', frame)

        # finalizar a gravação da webcam
        tecla = cv2.waitKey(1)
        if tecla == 27:
            break
webcam.release()
