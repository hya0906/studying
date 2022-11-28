import cv2

from media_func import *
from test_face import *

#flag=True
if __name__=="__main__":
    cap = cv2.VideoCapture(0)
    #cap.set(cv2.CAP_PROP_FPS, 100)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    c = 0

    print("시작")
    with mp_hands.Hands(
            model_complexity=0,
            max_num_hands=2,
            min_detection_confidence=0.9,
            min_tracking_confidence=0.8) as hands:
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                continue

            image.flags.writeable = False
            # 이미지 상하좌우 반전(1 : 좌우 반전, 0 : 상하 반전)
            image = cv2.flip(image, 1)
            if c%6 == 0:
                flag = recognize_faces(image)  ############
                c=0
            else:
                draw_faces(image)
            c+=1
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(image)


            if flag == True:

                # 손가락이 접힘 = true
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # 프레임 확인
                frame_count += 1
                if time.time() - frame_time >= 1:
                    # print("frame : %d" %(frame_count/(time.time()-frame_time)))
                    frame_result = int(frame_count / (time.time() - frame_time))
                    frame_count = 0
                    frame_time = time.time()
                cv2.putText(image, str(frame_result), (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (20, 255, 155), 3, 5)

                if results.multi_hand_landmarks:
                    # 손가락 숫자 확인
                    hands_counts, fingers_statuses = count_hand_finger(results)

                    # 손의 정적 제스처 탐지
                    gesture = hands_static_gesture(results, fingers_statuses, hands_counts)

                    # 손의 정적 제스처를 저장
                    update_static_gesture(gesture)

                    # 펜 모드 확인
                    check_pen_mode(gesture)

                    # 손의 동적 제스처 탐지
                    action = hands_dynamic_gesture(results)

                    imgae = hand_drawing(image, results, gesture, action)

                    # print(RIGHT_static_gesture[-1])
                    menu_update(action, hands_counts)
                    # print(hands_counts)
                    # print(fingers_statuses)
                    # print(gesture)
                    # print(action)
                    # print(menu_sta)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #image = menu_drawing(image)
            cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                print(count)
                break
            #cv2.waitKey(20)
    cap.release()