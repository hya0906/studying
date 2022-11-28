import cv2
import mediapipe as mp
import numpy as np
import math
import time
from scipy import interpolate
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
width = 1280
height = 720
depth = 1000

# n_프레임 동안의 정적 제스처 저장 리스트
# n_frame 값 지정 : 총 2초가 되도록 설정 (ex. 프레임이 20 나옴 -> 40으로 설정)
n_frame = 40
mouse_percent = 0.6
click_percent = 0.5
point_percent = 0.8
dial_percent = 0.8
num_percent = 0.8
RIGHT_static_gesture = []
LEFT_static_gesture = []
RIGHT_none_frame = 0
LEFT_none_frame = 0

# 초당 frame을 확인하기 위한 변수
frame_count = 0
frame_time = time.time()
frame_result = 0

# 마우스 포인트
left_mouse_point = [0, 0]
right_mouse_point = [0, 0]
left_mouse_clicked = []
right_mouse_clicked = []

# point
left_point = []
right_point = []

# dial
left_degree = []
right_degree = []
dial_start_degree = {'RIGHT': None, 'LEFT': None}
dial_num = {'RIGHT': 0, 'LEFT': 0}
dial_none_count = {'RIGHT': 0, 'LEFT': 0}

dial_next_points = {'RIGHT': [], 'LEFT': []}

dial_points_statuses = {'RIGHT': [0, 0, 0, 0], 'LEFT': [0, 0, 0, 0]}

dial_reset_point = {'RIGHT': None, 'LEFT': None}

# pen
pen_gesture_count = {'RIGHT': 0, 'LEFT': 0}
pen_mode = {'RIGHT': 'OFF', 'LEFT': 'OFF'}
pen_point = {'RIGHT': None, 'LEFT': None}
pen_pre_point = {'RIGHT': None, 'LEFT': None}
p2p_min_len = 100
line_x = ""
line_y = ""

# count
action_none_num = {'RIGHT': 0, 'LEFT': 0}
action_last = {'RIGHT' : '', 'LEFT' : ''}

# menu
# menu_sta : off, light, memo, tv, air, exit
menu_sta = {"main": 'off', "sub": "off"}
main_list = ["light", "memo", "tv", "air", "exit"]
main_num = 0
light_level = 3
tv_sta = {"volume": 0, "channel": 0}
air_cleaner = "off"
air_cleaner_mod = ["off", "auto", "pet", "sleep"]
exit_system = 0
memo_list = []
memo_open = "off"

#실행 수 저장
count = {"left": 0, "right": 0, "up": 0, "down": 0, "left_click": 0, "right_click": 0, "plus": 0, "minus": 0,
         "dial_reset": 0}


def xy_len(p1, p2):
    a = p1.x - p2.x
    a = a * width
    b = p1.y - p2.y
    b = b * height
    c = math.sqrt((a * a) + (b * b))
    return c


def get_angle(a, b):
    # 내적 공식을 이용한 두 벡터 사이의 각 구하기(np 활용)
    # np.dot : 행렬의 곱
    inner = np.dot(a, b)
    # norm 이란 벡터의 크기를 측정하는 방법(함수)이다.
    # 아래 식은 두 벡터의 절대값을 곱한다.
    vec_size = np.linalg.norm(a) * np.linalg.norm(b)
    angle = np.arccos(inner / vec_size)
    angle = angle / np.pi * 180
    return angle


def count_hand_finger(results):
    # 손가락이 접힘 = true
    fingers_statuses = {'RIGHT_THUMB': False, 'RIGHT_INDEX': False, 'RIGHT_MIDDLE': False, 'RIGHT_RING': False,
                        'RIGHT_PINKY': False, 'LEFT_THUMB': False, 'LEFT_INDEX': False, 'LEFT_MIDDLE': False,
                        'LEFT_RING': False, 'LEFT_PINKY': False}
    # -1 : 손이 탐지 x
    hands_count = {'LEFT': -1, 'RIGHT': -1}
    # 엄지는 손 모양에 따라 마디의 각도가 크게 변해서 2개 마디에서의 각도를 모두 확인함
    hand_landmarks_index = [
        [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_MCP],
        [mp_hands.HandLandmark.THUMB_IP, mp_hands.HandLandmark.THUMB_MCP, mp_hands.HandLandmark.THUMB_CMC],
        [mp_hands.HandLandmark.INDEX_FINGER_DIP, mp_hands.HandLandmark.INDEX_FINGER_PIP,
         mp_hands.HandLandmark.INDEX_FINGER_MCP],
        [mp_hands.HandLandmark.MIDDLE_FINGER_DIP, mp_hands.HandLandmark.MIDDLE_FINGER_PIP,
         mp_hands.HandLandmark.MIDDLE_FINGER_MCP],
        [mp_hands.HandLandmark.RING_FINGER_DIP, mp_hands.HandLandmark.RING_FINGER_PIP,
         mp_hands.HandLandmark.RING_FINGER_MCP],
        [mp_hands.HandLandmark.PINKY_DIP, mp_hands.HandLandmark.PINKY_PIP, mp_hands.HandLandmark.PINKY_MCP]]

    finger_name = ["THUMB", "THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]

    for hand_index, hand_info in enumerate(results.multi_handedness):
        # 왼손, 오른손 파악(hand_label)
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]

        hands_count[hand_label.upper()] = 5

        for num, i in enumerate(hand_landmarks_index):
            # 랜드마크의 크기는 0~1임으로 너무 작기 때문에 실제 위치로 계산(width, height, depth 곱)
            # 상단 랜드마크
            x1 = int(hand_landmarks.landmark[i[0]].x * width)
            y1 = int(hand_landmarks.landmark[i[0]].y * height)
            z1 = int(hand_landmarks.landmark[i[0]].z * depth)

            # 중간 랜드마크
            x2 = int(hand_landmarks.landmark[i[1]].x * width)
            y2 = int(hand_landmarks.landmark[i[1]].y * height)
            z2 = int(hand_landmarks.landmark[i[1]].z * depth)

            # 하단 랜드마크
            x3 = int(hand_landmarks.landmark[i[2]].x * width)
            y3 = int(hand_landmarks.landmark[i[2]].y * height)
            z3 = int(hand_landmarks.landmark[i[2]].z * depth)

            # 중간 -> 상단 벡터
            vec1_x = x1 - x2
            vec1_y = y1 - y2
            vec1_z = z1 - z2
            vec1 = np.array([vec1_x, vec1_y, vec1_z])

            # 중간 -> 하단 벡터
            vec2_x = x3 - x2
            vec2_y = y3 - y2
            vec2_z = z3 - z2
            vec2 = np.array([vec2_x, vec2_y, vec2_z])

            # 각도를 구함
            angle = get_angle(vec1, vec2)

            # 손가락 상태를 업데이트
            # 엄지와 소지의 각도 변화가 크지 않아서 따로 처리
            # 검지, 중지, 약지
            if (num != 0 and num != 1 and num != 5):
                if angle < 110:
                    fingers_statuses[hand_label.upper() + "_" + finger_name[num]] = True
                    hands_count[hand_label.upper()] -= 1
            # 엄지, 소지
            else:
                if angle < 150:
                    # 엄지가 2번 카운트 되는 것을 방지하기 위한 if문
                    if (fingers_statuses[hand_label.upper() + "_" + finger_name[num]] == False):
                        fingers_statuses[hand_label.upper() + "_" + finger_name[num]] = True
                        hands_count[hand_label.upper()] -= 1

    return hands_count, fingers_statuses


def hands_static_gesture(results, finger_statuses, hands_counts):
    gesture = {'LEFT': "", 'RIGHT': ""}

    # 손의 수만큼 for문 동작(양손이기 때문에 2회 실시)
    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        min_len1 = xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_DIP]) // 2 * 3
        min_len2 = xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP])
        pen_min_len = xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                             hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]) // 4
        update_point(hand_label.upper(), hand_landmarks)
        gesture[hand_label.upper()] = "none"
        dial_none_count[hand_label.upper()] += 1
        # 메모장 켜셔 있으면 현재 팬촉 위치 업데이트
        if pen_mode[hand_label.upper()] != "off":
            pen_point[hand_label.upper()] = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        # 정적 제스처 파악
        if dial_none_count[hand_label.upper()] == n_frame // 2:
            if hand_label.upper() == "LEFT":
                left_degree.clear()
                dial_next_points["LEFT"].clear()

            else:
                right_degree.clear()
                dial_next_points["RIGHT"].clear()
        if hands_counts[hand_label.upper()] == 2:
            if (finger_statuses[hand_label.upper() + "_INDEX"] == False) and (
                    finger_statuses[hand_label.upper() + "_MIDDLE"] == False):
                # 중지와 검지가 붙어있으면
                if xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                          hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < min_len1:
                    gesture[hand_label.upper()] = "point"
                else:
                    gesture[hand_label.upper()] = "mouse"
                    # 마우스 포인트 위치 저장
                    mouse_point(hand_label.upper(), hand_landmarks)



        elif hands_counts[hand_label.upper()] == 1:
            # 중지만 펴져 있는 경우
            if finger_statuses[hand_label.upper() + "_MIDDLE"] == False:
                if hand_label.upper() == "RIGHT":
                    gesture[hand_label.upper()] = "left_click"
                else:
                    gesture[hand_label.upper()] = "right_click"
                # 마우스 포인트 위치 저장
                mouse_point(hand_label.upper(), hand_landmarks)

            elif finger_statuses[hand_label.upper() + "_INDEX"] == False:
                if hand_label.upper() == "RIGHT":
                    gesture[hand_label.upper()] = "right_click"
                else:
                    gesture[hand_label.upper()] = "left_click"
                # 마우스 포인트 위치 저장
                mouse_point(hand_label.upper(), hand_landmarks)

        # elif hands_counts[hand_label.upper()] == 3:
        #     if (finger_statuses[hand_label.upper() + "_THUMB"] == False) and (
        #             finger_statuses[hand_label.upper() + "_MIDDLE"] == False) and (
        #             finger_statuses[hand_label.upper() + "_INDEX"] == False):
        #         if xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        #                   hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < min_len2:
        #             gesture[hand_label.upper()] = "dial"
        #             update_degree(hand_label, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
        #                           hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
        #             dial_none_count[hand_label.upper()] = 0

        # 손가락 숫자가 3~5 -> 펜 모드, dial 확인

        elif (finger_statuses[hand_label.upper() + "_THUMB"] == False) and (
                finger_statuses[hand_label.upper() + "_MIDDLE"] == False) and (
                finger_statuses[hand_label.upper() + "_INDEX"] == False) and (
                finger_statuses[hand_label.upper() + "_RING"] == True):
            if xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                      hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]) < min_len2:
                gesture[hand_label.upper()] = "dial"
                update_degree(hand_label, hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                              hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP])
                dial_none_count[hand_label.upper()] = 0
        # pen 제스처
        elif xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]) < pen_min_len:
            pen_gesture_count[hand_label.upper()] += 1

            gesture[hand_label.upper()] = "pen"
        # pen off 제스처
        elif xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP],
                    hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]) < min_len1:
            gesture[hand_label.upper()] = "pen_off"
            pen_gesture_count[hand_label.upper()] -= 1

    return gesture


def hands_dynamic_gesture(results):
    action = {'LEFT': "", 'RIGHT': ""}
    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        point_min_len = xy_len(hand_landmarks.landmark[mp_hands.HandLandmark.WRIST],
                               hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP])
        action[hand_label.upper()] = "none"
        if hand_label.upper() == "LEFT":
            static_gesture = LEFT_static_gesture
        else:
            static_gesture = RIGHT_static_gesture
        # 동적 제스처를 실행하기 위한 기본적인 조건
        # 1. 40프레임 동안의 정적 제스처가 인식됨
        if len(static_gesture) == n_frame:
            # 마지막 제스처가 클릭 일 경우
            if "click" in static_gesture[n_frame - 1]:
                # 마우스 동적 제스처 탐지
                check_mouse = 0
                for i in range(n_frame // 2):
                    if static_gesture[i] == "mouse":
                        check_mouse += 1

                if check_mouse > int((n_frame // 2) * mouse_percent):
                    check_click = 0
                    for i in range(n_frame // 4):
                        if "click" in static_gesture[n_frame - 1 - i]:
                            check_click += 1
                    if check_click > int((n_frame // 4) * click_percent):
                        action[hand_label.upper()] = static_gesture[n_frame - 1]

                        if hand_label.upper() == "RIGHT":
                            right_mouse_clicked.append(
                                [right_mouse_point[0], right_mouse_point[1], action[hand_label.upper()]])
                            RIGHT_static_gesture.clear()
                        else:
                            left_mouse_clicked.append(
                                [left_mouse_point[0], left_mouse_point[1], action[hand_label.upper()]])
                            LEFT_static_gesture.clear()


            elif static_gesture[n_frame - 1] == "point":
                # 포인트 동적 제스처 탐지
                check_point = 0
                point_action = ""
                for i in range(n_frame):
                    if static_gesture[i] == "point":
                        check_point += 1
                if check_point > int((n_frame) * point_percent):
                    if hand_label.upper() == "LEFT":
                        point = left_point
                    else:
                        point = right_point
                    # x축 이동 확인
                    if abs(point[0][0] - point[n_frame - 1][0]) > point_min_len:
                        if point[0][0] > point[n_frame - 1][0]:
                            point_action = "left"
                        else:
                            point_action = "right"
                    if abs(point[0][1] - point[n_frame - 1][1]) > point_min_len:
                        if point[0][1] > point[n_frame - 1][1]:
                            point
                            point_action += "up"
                        else:
                            point_action += "down"
                if point_action != "":
                    action[hand_label.upper()] = point_action
                    if hand_label.upper() == "RIGHT":
                        RIGHT_static_gesture.clear()
                    else:
                        LEFT_static_gesture.clear()
            elif static_gesture[n_frame - 1] == "dial":
                check_dial = 0
                # 예민하게 반응하기 위해 40 프레임 중 4분의 1만을 이용하여 판단
                for i in range(n_frame // 4):
                    if static_gesture[n_frame - 1 - i] == "dial":
                        check_dial += 1
                if check_dial > int((n_frame // 4) * dial_percent):

                    if hand_label.upper() == "LEFT":
                        degree = left_degree
                        start_degree = dial_start_degree["LEFT"]
                    else:
                        degree = right_degree
                        start_degree = dial_start_degree["RIGHT"]

                    if degree[-1] > start_degree + 30 and dial_points_statuses[hand_label.upper()][1] != 1:
                        dial_points_statuses[hand_label.upper()][1] = 1
                        dial_num[hand_label.upper()] -= 1
                        action[hand_label.upper()] = "minus"

                    if degree[-1] > start_degree + 60 and dial_points_statuses[hand_label.upper()][0] != 1:
                        dial_points_statuses[hand_label.upper()][0] = 1
                        dial_num[hand_label.upper()] -= 1
                        action[hand_label.upper()] = "minus"

                    if degree[-1] < start_degree and dial_points_statuses[hand_label.upper()][1] == 1:
                        dial_points_statuses[hand_label.upper()][1] = 0
                        dial_points_statuses[hand_label.upper()][0] = 0
                        action[hand_label.upper()] = "dial_reset"

                    if degree[-1] > start_degree and dial_points_statuses[hand_label.upper()][2] == 1:
                        dial_points_statuses[hand_label.upper()][2] = 0
                        dial_points_statuses[hand_label.upper()][3] = 0
                        action[hand_label.upper()] = "dial_reset"

                    if degree[-1] < start_degree - 30 and dial_points_statuses[hand_label.upper()][2] != 1:
                        dial_points_statuses[hand_label.upper()][2] = 1
                        dial_num[hand_label.upper()] += 1
                        action[hand_label.upper()] = "plus"

                    if degree[-1] < start_degree - 60 and dial_points_statuses[hand_label.upper()][3] != 1:
                        dial_points_statuses[hand_label.upper()][3] = 1
                        dial_num[hand_label.upper()] += 1
                        action[hand_label.upper()] = "plus"

            if action[hand_label.upper()] != "none":
                action_none_num[hand_label.upper()] = 0
                action_last[hand_label.upper()] = action[hand_label.upper()]

            else:
                action_none_num[hand_label.upper()] += 1
                if action_none_num[hand_label.upper()] > n_frame // 2:
                    action[hand_label.upper()] = "count"
        #실행 횟수 저장
        # if action[hand_label.upper()] != "none" and action[hand_label.upper()] != "count":
        #     print(action[hand_label.upper()])
        #     print("횟수 : ")
        #     count[action[hand_label.upper()]] += 1
        #     print(count[action[hand_label.upper()]])
    return action


def mouse_point(hand_label, hand_landmarks):
    point1_x = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].x
    point1_y = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP].y
    point2_x = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x
    point2_y = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y
    point3_x = (point1_x + point2_x) / 2
    point3_y = (point1_y + point2_y) / 2
    point_x = (2 * point3_x - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x) * width
    point_y = (2 * point3_y - hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].y) * height
    if hand_label == "LEFT":
        left_mouse_point[0] = int(point_x)
        left_mouse_point[1] = int(point_y)
    else:
        right_mouse_point[0] = int(point_x)
        right_mouse_point[1] = int(point_y)


def update_degree(hand_label, point1, point2):
    # 손의 위치마다 다음 포인트를 표시하기 위해 매 프레임마다 포인트들 초기화

    center_x = (point1.x + point2.x) / 2
    center_y = (point1.y + point2.y) / 2
    l = math.sqrt((center_x - point1.x) ** 2 + (center_y - point1.y) ** 2)

    rad = math.atan2(point1.y - center_y, point1.x - center_x) * (-1)
    deg = int((rad * 180) / math.pi)

    if hand_label.upper() == "LEFT":
        dial_next_points['LEFT'].clear()
        dial_reset_point['LEFT'] = ""
        # 시작 각도 저장
        if len(left_degree) == 0:
            dial_start_degree["LEFT"] = deg

            # 다음 포인트들 계산
        for i in range(-2, 3):

            rad = np.pi / 180 * (dial_start_degree["LEFT"] + 30 * i)
            x = l * math.cos(-rad) + center_x
            y = l * math.sin(-rad) + center_y
            x = int(x * width)
            y = int(y * height)
            if i == 0:
                dial_reset_point['LEFT'] = [x, y]
            else:
                dial_next_points['LEFT'].insert(0, [x, y])
        left_degree.append(deg)

        if len(left_degree) == (n_frame // 4) + 1:
            left_degree.pop(0)

    else:
        dial_next_points['RIGHT'].clear()
        dial_reset_point['RIGHT'] = ""
        if len(right_degree) == 0:
            dial_start_degree["RIGHT"] = deg

        # 다음 포인트들 계산
        for i in range(-2, 3):

            rad = np.pi / 180 * (dial_start_degree["RIGHT"] + 30 * i)
            x = l * math.cos(-rad) + center_x
            y = l * math.sin(-rad) + center_y
            x = int(x * width)
            y = int(y * height)
            if i == 0:
                dial_reset_point['RIGHT'] = [x, y]
            else:
                dial_next_points['RIGHT'].insert(0, [x, y])
        # print(dial_next_points['RIGHT'])
        right_degree.append(deg)

        # print(right_next_points)
        if len(right_degree) == (n_frame // 4) + 1:
            right_degree.pop(0)
    # print(right_degree)


# n 프레임 만큼 업데이트
def update_static_gesture(gesture):
    global RIGHT_none_frame
    global LEFT_none_frame

    # n_frame 동안 손가락이 탐지되지 않거나 제스처가 인식되지 않으면 리스트를 초기화
    # if gesture['LEFT'] != 'none':
    #     LEFT_static_gesture.append(gesture['LEFT'])
    #     if len(LEFT_static_gesture) == n_frame + 1:
    #         LEFT_static_gesture.pop(0)
    #     LEFT_none_frame = 0
    # else:
    #     if LEFT_none_frame == 0:
    #
    #     LEFT_none_frame += 1
    #     if LEFT_none_frame == n_frame:
    #         LEFT_static_gesture.clear()
    #
    # if gesture['RIGHT'] != 'none':
    #     RIGHT_static_gesture.append(gesture['RIGHT'])
    #     if len(RIGHT_static_gesture) == n_frame + 1:
    #         RIGHT_static_gesture.pop(0)
    #     RIGHT_none_frame = 0
    # else:
    #     RIGHT_none_frame += 1
    #     if RIGHT_none_frame == n_frame:
    #         RIGHT_static_gesture.clear()

    LEFT_static_gesture.append(gesture['LEFT'])
    if len(LEFT_static_gesture) == n_frame + 1:
        LEFT_static_gesture.pop(0)

    RIGHT_static_gesture.append(gesture['RIGHT'])
    if len(RIGHT_static_gesture) == n_frame + 1:
        RIGHT_static_gesture.pop(0)


# n 프레임 동안의 포인트 업데이트
def update_point(hand_label, hand_landmarks):
    point_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width)
    point_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)
    if hand_label == "LEFT":
        left_point.append([point_x, point_y])
        if len(left_point) == n_frame + 1:
            left_point.pop(0)
    else:
        right_point.append([point_x, point_y])
        if len(right_point) == n_frame + 1:
            right_point.pop(0)


# 화면에 손 관련 그림
def hand_drawing(image, results, gesture, action):
    for hand_index, hand_info in enumerate(results.multi_handedness):
        hand_label = hand_info.classification[0].label
        hand_landmarks = results.multi_hand_landmarks[hand_index]
        #손의 랜드마크를 찍어줌
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        #탐지한 액션 출력
        if action_last[hand_label.upper()] != "none":
            if action_none_num[hand_label.upper()] < n_frame * 2:
                if hand_label.upper() == "LEFT":
                    temp = "LEFT:" + action_last[hand_label.upper()]
                    image = cv2.putText(image, temp, (1000,100), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (20, 255, 155), 3, 5)
                else:
                    temp = "RIGHT:" + action_last[hand_label.upper()]
                    image = cv2.putText(image, temp, (1000,50), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (20, 255, 155), 3, 5)
        # 마우스 커스 위치를 보여줌
        if (gesture[hand_label.upper()] == "mouse") or ("click" in gesture[hand_label.upper()]):
            if hand_label.upper() == "LEFT":
                image = cv2.circle(image, (left_mouse_point[0], left_mouse_point[1]), 1, (20, 255, 155), 5)
            else:
                image = cv2.circle(image, (right_mouse_point[0], right_mouse_point[1]), 1, (20, 255, 155), 5)
        elif gesture[hand_label.upper()] == "point":
            image = cv2.circle(image, (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width),
                                       int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)),
                               1, (20, 255, 155), 5)
        elif (gesture[hand_label.upper()] == "dial"):
            temp = "LEFT:" + str(dial_num["LEFT"]) + "//RIGHT:" + str(dial_num["RIGHT"])
            image = cv2.putText(image, temp, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                1, (20, 255, 155), 3, 5)
            image = cv2.line(image, (int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * width),
                                     int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * height)), (
                                 int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].x * width),
                                 int(hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP].y * height)), (255, 0, 0),
                             5)

            image = cv2.circle(image,
                               (dial_reset_point[hand_label.upper()][0], dial_reset_point[hand_label.upper()][1]),
                               1, (20, 255, 155), 5)
            for i in range(4):
                if dial_points_statuses[hand_label.upper()][i] == 0:
                    color = (0, 0, 255)
                else:
                    color = (255, 0, 0)
                image = cv2.circle(image, (
                    dial_next_points[hand_label.upper()][i][0], dial_next_points[hand_label.upper()][i][1]), 1, color,
                                   5)

    # 클릭했던 위치를 그림
    # 왼손
    # for i in left_mouse_clicked:
    #     if i[2] == "left_click":
    #         image = cv2.circle(image, (i[0], i[1]), 1, (0, 0, 0), 5)
    #     else:
    #         image = cv2.circle(image, (i[0], i[1]), 1, (255, 255, 255), 5)
    # # 오른손
    # for i in right_mouse_clicked:
    #     if i[2] == "left_click":
    #         image = cv2.circle(image, (i[0], i[1]), 1, (255, 0, 0), 5)
    #     else:
    #         image = cv2.circle(image, (i[0], i[1]), 1, (0, 0, 255), 5)
    return image


def check_pen_mode(gesture):
    global line_x
    global line_y
    for hand, count in pen_gesture_count.items():
        # 펜 모드 on
        if count == int(n_frame * 1.5):
            if pen_mode[hand] == "OFF":
                temp = str(time.localtime().tm_year) + "_" + str(time.localtime().tm_mon) + "_" + str(
                    time.localtime().tm_mday) + "_" + str(time.localtime().tm_hour) + "_" + str(
                    time.localtime().tm_min) + ".jpg"
                # temp = hand + ".jpg"
                pen_mode[hand] = temp
                pen_note = np.zeros((height, width), np.uint8)
                cv2.imwrite(temp, pen_note)
        # 펜 모드 off
        elif count == -1:
            if pen_mode[hand] != "OFF":
                pen_interpolate(pen_mode[hand])
                cv2.destroyWindow(pen_mode[hand])
                pen_mode[hand] = "OFF"
            pen_gesture_count[hand] = 0
        # 펜 count가 계속 커지는 것을 방지(82 --> 81)
        elif count == (n_frame * 2) + 2:
            pen_gesture_count[hand] = (n_frame * 2) + 1

    for hand, mode in pen_mode.items():
        # 메모장이 켜져 있음
        if mode != "OFF":
            x = int(pen_point[hand].x * width)
            y = int(pen_point[hand].y * height)
            # 제스처가 pen -> 이미지를 새롭게 업데이트(현재 point에 점을 찍음)
            if gesture[hand] == 'pen':
                pen_note = cv2.imread(mode, cv2.IMREAD_ANYDEPTH)

                pen_note = cv2.circle(pen_note, (x, y), 3, 150, -1)

                # 점으로 찍히는 문제 보완 + 선 정리
                if pen_pre_point[hand] != None:
                    pen_note = cv2.line(pen_note, (x, y),
                                        (int(pen_pre_point[hand].x * width), int(pen_pre_point[hand].y * height)),
                                        150, 5)
                pen_pre_point[hand] = pen_point[hand]
                line_x += str(x) + ","
                line_y += str(y) + ","
                cv2.imwrite(mode, pen_note)
            else:
                # 바로 직전에 pen 입력이 있었음
                if pen_pre_point[hand] != None:
                    line_x = line_x[: -1] + "/"
                    line_y = line_y[: -1] + "/"
                pen_pre_point[hand] = None
            pen_note = cv2.imread(mode, cv2.IMREAD_ANYDEPTH)
            pen_note = cv2.circle(pen_note, (x, y), 5, 255, -1)
            cv2.imshow(mode, pen_note)


def pen_interpolate(mode):
    global line_x
    global line_y
    line_x = line_x[:-1].split("/")
    line_y = line_y[:-1].split("/")
    # 각 선을 정수형태의 리스트로 저장
    for i in range(0, len(line_x)):
        line_x[i] = line_x[i].split(",")
        line_x[i] = list(map(int, line_x[i]))

        line_y[i] = line_y[i].split(",")
        line_y[i] = list(map(int, line_y[i]))
    pen_note = np.zeros((height, width), np.uint8)

    for i in range(0, len(line_x)):
        if len(line_x[i]) > 3:
            l = len(line_x[i])

            t = np.linspace(0, 1, l - 2, endpoint=True)
            t = np.append([0, 0, 0], t)
            t = np.append(t, [1, 1, 1])

            tck = [t, [line_x[i], line_y[i]], 3]
            u3 = np.linspace(0, 1, (max(l * 2, 70)), endpoint=True)

            out = interpolate.splev(u3, tck)

            for j in range(0, len(out[0])):
                pen_note = cv2.circle(pen_note, (int(out[0][j]), int(out[1][j])), 3, 150, -1)
                if j != 0:
                    pen_note = cv2.line(pen_note, (int(out[0][j]), int(out[1][j])),
                                        (int(out[0][j - 1]), int(out[1][j - 1])),
                                        250, 5)
        else:
            for j in range(0, len(line_x[i])):
                pen_note = cv2.circle(pen_note, (line_x[i][j], line_y[i][j]), 3, 150, -1)
                if j != 0:
                    pen_note = cv2.line(pen_note, (line_x[i][j], line_y[i][j]),
                                        (line_x[i][j - 1], line_y[i][j - 1]),
                                        250, 5)
    #모델 입력을 위한 반전
    pen_note = 255 - pen_note
    cv2.imwrite("md_" + mode, pen_note)


def menu_update(action, hands_counts):
    global menu_sta
    global main_num
    global light_level
    global air_cleaner
    global exit_system
    global memo_open

    for hand, hand_action in action.items():
        if hand_action != "":
            if hand == "RIGHT":
                static_gesture = RIGHT_static_gesture
                mouse_clicked = right_mouse_clicked
            else:
                static_gesture = LEFT_static_gesture
                mouse_clicked = left_mouse_clicked
            # 메뉴 off에서 down 액션 들어오면 메인 메뉴 ON
            if menu_sta["main"] == "off":
                if hand_action == "down":
                    main_num = 0
                    menu_sta["main"] = main_list[main_num]
            # 메인 메뉴 on
            elif menu_sta["main"] != "off":
                # 서브 메뉴 off
                if menu_sta["sub"] == "off":
                    if hand_action == "up":
                        menu_sta["main"] = "off"
                    elif hand_action == "left":
                        main_num -= 1
                        if main_num < 0:
                            main_num = 0
                        menu_sta["main"] = main_list[main_num]
                    elif hand_action == "right":
                        main_num += 1
                        if main_num > 4:
                            main_num = 4
                        menu_sta["main"] = main_list[main_num]
                    elif hand_action == "down":
                        menu_sta["sub"] = "on"
                # 서브 메뉴 on
                else:
                    if hand_action == "up":
                        menu_sta["sub"] = "off"
                        if memo_open != "off":
                            cv2.destroyWindow(memo_open)
                            memo_open = "off"
                    # light 서브 메뉴 조작
                    if menu_sta["main"] == "light":
                        if hand_action == "left":
                            light_level -= 1
                            if light_level < 1:
                                light_level = 1
                        elif hand_action == "right":
                            light_level += 1
                            if light_level > 5:
                                light_level = 5
                        elif hand_action == "count" and static_gesture[-1] != "point":
                            if hands_counts[hand] != 0:
                                light_level = hands_counts[hand]
                                # 연속 입력을 막기 위해 count 관련 none_num 초기화
                                action_none_num[hand] = 0

                    # tv 서브 메뉴 조작
                    elif menu_sta["main"] == "tv":
                        # 오른손 -> 채널 조작
                        if hand == "RIGHT":
                            if hand_action == "plus":
                                tv_sta["channel"] += 1
                                if tv_sta["channel"] > 30:
                                    tv_sta["channel"] = 30
                            elif hand_action == "minus":
                                tv_sta["channel"] -= 1
                                if tv_sta["channel"] < 0:
                                    tv_sta["channel"] = 0
                        # 왼손 -> 볼륨 조작
                        else:
                            if hand_action == "plus":
                                tv_sta["volume"] += 1
                                if tv_sta["volume"] > 30:
                                    tv_sta["volume"] = 30
                            elif hand_action == "minus":
                                tv_sta["volume"] -= 1
                                if tv_sta["volume"] < 0:
                                    tv_sta["volume"] = 0
                    # 공기청정기 서브 메뉴 조작
                    elif menu_sta["main"] == "air":
                        if hand_action == "count":
                            if hands_counts[hand] != 0 and hands_counts[hand] != 5:
                                air_cleaner = air_cleaner_mod[hands_counts[hand] - 1]
                                # 연속 입력을 막기 위해 count 관련 none_num 초기화
                                action_none_num[hand] = 0
                                menu_sta["sub"] = "off"
                    # 외출 모드 메뉴 조작
                    elif menu_sta["main"] == "exit":
                        if hand_action == "down":
                            exit_system = 1

                    elif menu_sta["main"] == "memo":
                        if hand_action == "left_click":
                            for i, memo_name in enumerate(memo_list):
                                if mouse_clicked[-1][0] > 590 and mouse_clicked[-1][0] < 680:
                                    if (mouse_clicked[-1][1] > 65 + (i + 1) * 50) and (mouse_clicked[-1][1] < 95 + (i + 1) * 50):
                                        memo_open = memo_name
                                        print(memo_name)
                                        memo_jpg = cv2.imread(memo_name)
                                        cv2.imshow(memo_name, memo_jpg)
                        if hand_action == "right_click" and memo_open != "off":
                            cv2.destroyWindow(memo_open)
                            memo_open = "off"


def get_memo_list():
    global memo_list
    memo_list = []
    # 현재 코드의 위치 출력(절대 경로)
    root_dir = os.path.abspath(__file__)
    temp = root_dir.split('\\')
    root_dir = '\\'.join(s for s in temp[0:-1])
    files = os.listdir(root_dir)
    jpg_files = []
    for file in files:
        if ('.jpg' in file) and ('md' in file):
            memo_list.append(file)


def menu_drawing(image):
    # 화면 밝기 조절
    image = cv2.add(image, ((light_level - 3) * 30, (light_level - 3) * 30, (light_level - 3) * 30, 0))

    if menu_sta["main"] != "off":
        temp = cv2.imread("menu.png")
        menu_jpg = cv2.resize(temp, (1280, 100))
        image[0:100, 0:1280] = menu_jpg
        # 메인 메뉴에서 현재 위치를 표시
        if menu_sta["main"] == "light":
            image = cv2.line(image, (230, 90), (330, 90), (0, 0, 0), 5)
        elif menu_sta["main"] == "memo":
            image = cv2.line(image, (415, 90), (515, 90), (0, 0, 0), 5)
        elif menu_sta["main"] == "tv":
            image = cv2.line(image, (605, 90), (705, 90), (0, 0, 0), 5)
        elif menu_sta["main"] == "air":
            image = cv2.line(image, (785, 90), (885, 90), (0, 0, 0), 5)
        elif menu_sta["main"] == "exit":
            image = cv2.line(image, (975, 90), (1075, 90), (0, 0, 0), 5)

        # 공기청정기 상태 표현
        image = cv2.putText(image, air_cleaner, (870, 30), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 1, 1)

        if menu_sta["sub"] == "on":
            if menu_sta["main"] == "light":
                for i in range(light_level):
                    image = cv2.circle(image, (240 + i * 20, 120), 5, (0, 0, 255), -1)

            elif menu_sta["main"] == "tv":
                text2 = "channel:" + str(tv_sta["channel"])
                text1 = "volume:" + str(tv_sta["volume"])
                image = cv2.putText(image, text1, (460, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 0), 2, 5)
                image = cv2.putText(image, text2, (660, 130), cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 0), 2, 5)
            elif menu_sta["main"] == "air":
                image = cv2.putText(image, "1:off 2:auto 3:pet 4:sleep", (630, 130), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 0), 2, 5)
            elif menu_sta["main"] == "exit":
                image = cv2.putText(image, "Down(system exit)", (930, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
                                    5)
            elif menu_sta["main"] == "memo":
                get_memo_list()
                for i, name in enumerate(memo_list):
                    image = cv2.putText(image, name, (100, 85 + (i + 1) * 50), cv2.FONT_HERSHEY_SIMPLEX,
                                           1, (128, 128, 128), 2, 5)
                    image = cv2.putText(image, "click", (600, 90 + (i + 1) * 50), cv2.FONT_HERSHEY_SIMPLEX,
                                           1, (128, 128, 128), 2, 5)
                    image = cv2.rectangle(image, (590, 65 + (i + 1) * 50), (680, 95 + (i + 1) * 50), (128, 128, 128), 1)
    return image
