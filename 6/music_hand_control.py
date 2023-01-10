
'''

   Program uzywa rozpoznawania gestu dłoni i przełącza podłączony program spotify, wykrywa urządzenie spotify
   W związku z tym utworzony został projekt deweloperski spotify i tam dodane konta,
   które musiały zaakceptować autoryzacje żeby móc nimi sterować.
   Następnym krokiem było zaimplementowanie rozpoznawania gestu dłoni i podłączenie pod gesty akcji spotify.
   
   Obsługa spotify w pliku: spotify.py
   Obsługa gestu dłoni i wykorzystanie w tym programu spotify: music_hand_control.py

    Jak przygotwac srodowisko do uruchomienia programu:
    zajrzeć do pliku: REQUIREMENTS.TXT

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)

    Dodatkowe załączniki: WIDEO: 2023-01-02 17-11-41.mkv

'''
import time

import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# For webcam input:
import spotify
cap = cv2.VideoCapture(0)
with mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

#dodanie licznika do włączenia akcji
  action_counter = 0
#dodanie licznika do możliwego uruchomienia następnej akcji
  next_action_counter = 0
  while cap.isOpened():
    success, image = cap.read()
    if not success:
      print("Ignoring empty camera frame.")
      # If loading a video, use 'break' instead of 'continue'.
      continue

    # To improve performance, optionally mark the image as not writeable to
    # pass by reference.
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if results.multi_hand_landmarks:

      stop_printing_for_action = True

      if(stop_printing_for_action):
          for hand_landmarks in results.multi_hand_landmarks:

            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())


            #czytanie położeń interesujących punktów wysokości palców
            INDEX_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y
            INDEX_MIDDLE = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP].y

            MIDDLE_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP].y
            MIDDLE_MIDDLE = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP].y

            RING_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP].y
            RING_MIDDLE = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_PIP].y

            PINKY_TIP = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP].y
            PINKY_MIDDLE = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_PIP].y

            if next_action_counter > 0:
                next_action_counter -= 1/30

#poniżej cała logika odpowiadająca za rozpoznawanie pozy dłoni
            if(INDEX_TIP < INDEX_MIDDLE
                and MIDDLE_TIP > MIDDLE_MIDDLE
                and RING_TIP > RING_MIDDLE
                and PINKY_TIP > PINKY_MIDDLE):

                action_counter += 1 / 30

                if action_counter > 3 and next_action_counter == 0:
                    print(
                        f'Next song')
                    spotify.next_song()
                    next_action_counter = 3

            elif (INDEX_TIP < INDEX_MIDDLE
                    and MIDDLE_TIP < MIDDLE_MIDDLE
                    and RING_TIP > RING_MIDDLE
                    and PINKY_TIP > PINKY_MIDDLE):

                action_counter += 1/30

                if action_counter > 3 and next_action_counter == 0:
                    spotify.previous_song()
                    print(
                        f'Previous song')
                    next_action_counter = 3

            elif (INDEX_TIP < INDEX_MIDDLE
                    and MIDDLE_TIP < MIDDLE_MIDDLE
                    and RING_TIP < RING_MIDDLE
                    and PINKY_TIP > PINKY_MIDDLE):

                action_counter += 1 / 30

                if action_counter > 3 and next_action_counter == 0:
                    spotify.pause_song()
                    print(
                        f'stop song')
                    next_action_counter = 3

            elif (INDEX_TIP < INDEX_MIDDLE
                    and MIDDLE_TIP < MIDDLE_MIDDLE
                    and RING_TIP < RING_MIDDLE
                    and PINKY_TIP < PINKY_MIDDLE):

                action_counter += 1 / 30

                if action_counter > 3 and next_action_counter == 0:
                    spotify.start_song()
                    print(
                        f'start song')
                    next_action_counter = 3

            else:
                action_counter = 0
                next_action_counter = 0
                pass

            if next_action_counter < 0:
                next_action_counter = 0

    # Flip the image horizontally for a selfie-view display.
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))
    if cv2.waitKey(5) & 0xFF == 27:
      break
cap.release()
