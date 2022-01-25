current_r, current_c, current_d = 7, 4, 0
current_room_map = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 1, 1, 1, 0, 1],
    [1, 0, 0, 1, 1, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 1, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
]

#로봇 청소기가 청소할 칸의 개수를 알고싶음
#모든 칸을 탐색하는 방법을 사용해서 내가 갈 수 있는 모든 칸을 세줘야 함->전부탐색, 모든칸 탐색->BFS!!

#1. 현재 위치를 청소한다. -청소한 위치를 기록해 두어야 한다.
#bfs를 구현 visited = [1,2,3] ->이걸 2차원으로
#0-청소안함 1-청소못함 2-청소함

# 북 동 남 서
dr = [-1, 0, 1, 0] #각각 방향이 나타내는 row의 변동 수를 적기
dc = [0, 1, 0, -1] #column이 각각 방향에 따라 어떻게 변하는지 적기

#2. 현재 위치에서 현재 방향을 기준으로 왼쪽 방향부터 차례대로 탐색을 진행한다
#->'방향'이라는 개념 ->위로가면 -1 아래 +1
#   r  c
#북 -1  0
#동  0  1
#남  1  0
#서  0 -1

# 방향 전환
def get_d_index_when_rotate_to_left(d):
    return (d + 3) % 4

#a.왼쪽 방향에 아직 청소하지 않은 공간이 존재한다면
# 그 방향으로 회전한 다음 한 칸을 전진하고 1번부터 진행한다.
#->'회전'이라는 개념 - 북 0 동 1 남 2 서 3
#북 왼 회전? 서 0->3
#동 왼 회전? 북 1->0
#남 왼 회전? 동 2->1
#서 왼 회전? 남 3->2

# 후진
def get_d_index_when_go_back(d):
    return (d + 2) % 4

#b. 왼쪽 방향에 청소할 공간이 없다면, 그 방향으로 회전하고
#2번으로 돌아간다.
#->현재 본 방향에서 청소할 곳이 없다면 다시 왼쪽으로 회전하라는 의미

#c. 네 방향 모두 청소가 되어있거나 벽인 경우에는 바라보는 방향을 유지한 채로
#한 칸 후진하고 2번으로 돌아간다.
#->모든 방향이 청소되어있다면 뒤로 한칸 후진해야함
# 북 뒤 돌기? 남 0 -> 2
# 동 뒤 돌기? 서 1 -> 3
# 남 뒤 돌기? 북 2 -> 0
# 서 뒤 돌기? 동 3 -> 1

#d. 네 방향 모두 청소가 되어있거나 벽이면서, 뒤쪽 방향이 벽이라
#후진도 할 수 없는 경우에는 작동을 멈춘다.

#청소한 칸의 총개수를 반환해 주면 된다.
def get_count_of_departments_cleaned_by_robot_vacuum(r, c, d, room_map):
    n = len(room_map)
    m = len(room_map[0])
    count_of_departments_cleaned = 1  # 청소하는 칸의 개수 - 이미 하나는 청소했다고 생각하고 1로 함
    room_map[r][c] = 2 #청소하면 room_map을 2로 업데이트시킴
    queue = list([[r, c, d]]) #모든 칸 탐색- BFS를 해야 함-현재의 위치와 방향을 전부 기록하고 어떻게 탐색할 지에 대해 고민을 해야하기 때문에

    # 큐가 비어지면 종료
    while queue: #맨 처음 방향은 d
        r, c, d = queue.pop(0)
        temp_d = d #방향을 계속 돌리면서 할 것이기 때문에 임시 변수로 만듬

        for i in range(4): #모든 방향에 대해서 회전하면서 갈 수 있는지 없는지 확인
            temp_d = get_d_index_when_rotate_to_left(temp_d) #왼쪽으로 한번 회전한 값
            #서쪽이면                0               -1 이 나올 것임
            new_r, new_c = r + dr[temp_d], c + dc[temp_d] #회전했으면 왼쪽에서 가는 방향도 알아야 함

            # a - 한칸 이동한 값인데 이 곳이 갈수 있는 곳인지 없는 곳인지 확인하기 위해서 판단함
            if 0 <= new_r < n and 0 <= new_c < m and room_map[new_r][new_c] == 0: #테이블의 범위 안에 위치가 들어가야 함/청소하지 않은 칸이어야 한다.
                #새로운 r값과 c값에 들어와서 청소함
                count_of_departments_cleaned += 1 #청소한 장소 추가
                room_map[new_r][new_c] = 2 #그 칸을 청소했다고 기록
                queue.append([new_r, new_c, temp_d]) #큐에다가 새로운 칸의 값을 추가->이동한 칸에서 다시 한번 탐색을 해줘야 하기 때문
                break         #위치,방향정보 모두 저장

            # c
            elif i == 3:  # 갈 곳이 없었던 경우 3까지 못찾으면 모두 청소가 되어있거나 벽이라는 의미->후진해야함
                #후진한 것의 변화량을 r과 c에 적용
                new_r, new_c = r + dr[get_d_index_when_go_back(temp_d)], c + dc[get_d_index_when_go_back(temp_d)] ##d를 temp_d로 변경
                queue.append([new_r, new_c, temp_d]) #위치,방향정보 저장 ##d를 temp_d로 변경

                # d
                if room_map[new_r][new_c] == 1:  # 뒤가 벽인 경우
                    return count_of_departments_cleaned


print(get_count_of_departments_cleaned_by_robot_vacuum(current_r, current_c, current_d, current_room_map))
#여러 힌트들과 구현사항에 대해서 어떻게 구현해줄지, 코드 안에서 어떻게 표현하면 될지 생각하고 BFS에 대한 개념도
#똑바로 잡혀있어야 풀수있다.