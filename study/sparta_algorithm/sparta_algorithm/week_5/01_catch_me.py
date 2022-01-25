from collections import deque

c = 11
b = 2

#코니의 위치 변화
#코니는 처음 위치에서 1초후 1만큼, 매초마다 이전 이동거리 + 1만큼 움직임
#즉 증가하는 속도가 1초마다 1씩 증가함.
#->속도 1 2 3 4 5 6 7
#->위치 3 4 6 9 13 18
#시간에 따라 더해주면 됨 1초면 1, 2초면 2, 3초면 3

#브라운의 위치 변화 -너무 무수한 갈래가 나옴. 경우의 수가 많고 쉽게 일반화 되지 않음
# -> 모든 경우의 수를 나열하는 것 필요 -> BFS
#잡았다! 라는 의미는 똑같은 시간에 똑같은 위치에 존재해야함.
#B-1, b+1, 2*B
#1-1. 2-1 = 1
#1-2. 2+1 = 3
#1-3. 2*2 = 4

#1-1-1. 1-1 = 0
#1-1-2. 1+1 = 2
#1-1-3. 1*2 = 2

#1-2-1. 3-1 = 2
#1-2-2. 3+1 = 4
#1-2-3. 3*2 = 6

#시간은 +1
#위치는 코니, 브라운도 값이 자유자재로 바뀜
#규칙적으로 증가하는 것을 저장하기 위한 자료구조 -> 배열
#자유자재로 변화하는 값을 저장하기 위한 자료구조 ->딕셔너리

#나는 각 시간마다 브라운이 갈 수 있는 위치를 저장하고 싶다.->이게 있어야 코니와 만날수있는지 없는지 확인할 수 있기 때문에
#-> 배열안에 딕셔너리 [{}]
def catch_me(cony_loc, brown_loc):
    time = 0 #시간개념 꼭 필요- 같은 시간대에 만나야 하기 때문
    queue = deque()  # 모든 경우의 수를 구하기 위한 큐 만듬 BFS
    queue.append((brown_loc, 0))  # 중요★ 위치와 시간을 동시에 저장
    visited=[{} for _ in range(200001)] #각 원소들은 시간과 장소를 저장해야하므로

#visited[위치][시간] = visited[3]에 5라는 키가 있냐? 라고 3 위치에 5초에 간 적 있었냐?
#0   1 -> visited[2] = {0: True}
#2 ->1 -> visited[1] = {1: True}
#    3 -> visited[3] = {1: True}
#    4 -> visited[4] = {1: True}

    # 0   1  2 -> visited[2] = {0: True, 2:True}#간략하게 설명위해 여기만 추가함
    # 2 ->1  0-> visited[1] = {1: True}
    #    3   2-> visited[3] = {1: True}
    #    4   3-> visited[4] = {1: True}
    #        4
    #        8
#    visited[0] = {
#        2:True
#    }
#    visited[1] = {  #visited[1][3] =존재 => 3 in visited[1]
#        1:True,
#        3:True,
#        4:True
#    }
#    visited[2] = {
#        0: True,
#        2: True,
#        3: True,
#        4: True,
#        8: True
#    }

    while cony_loc <= 200000:
        cony_loc += time #시간만큼 +1 +2 +3 +4
        if time in visited[cony_loc]:
            return time
        # if cony_loc >200000: #while조건문으로 사용함
        #     return -1
        for i in range(0, len(queue)):
            current_position, current_time = queue.popleft()

            #모든 경우의 수를 구하기 위해서 3개 다함
            new_time = current_time + 1
            new_position = current_position - 1
            if new_position>=0 and new_position <= 200000:
                queue.append((new_position, new_time))

            new_position = current_position + 1
            if 0 <= new_position <= 200000:
                queue.append((new_position, new_time))

            new_position = current_position * 2
            if 0 <= new_position <= 200000:
                queue.append((new_position, new_time))

        time += 1

    return -1


print(catch_me(c, b))  # 5가 나와야 합니다!