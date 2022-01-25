k = 4  # 말의 개수

chess_map = [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]
start_horse_location_and_directions = [
    [0, 0, 0],
    [0, 1, 0],
    [0, 2, 0],
    [2, 2, 2]
]
# 이 경우는 게임이 끝나지 않아 -1 을 반환해야 합니다!
# 동 서 북 남
# →, ←, ↑, ↓
dr = [0, 0, -1, 1]
dc = [1, -1, 0, 0]

#0->1 동서
#1->0 서동
#2->3 북남
#3->2 남북

def get_d_index_when_go_back(d):
    if d % 2 == 0:
        return d + 1
    else:
        return d - 1

#말은 순서대로 이동합니다 -> 말의 순서에 따라 반복문
#말이 쌓일 수 있습니다 -> 맵에 말이 쌓이는 걸 저장해놔야 됩니다.
#쌓인 순서대로 이동합니다 ->stack
#현재 맵에 어떻게 말이 쌓일지를 저장하기 위해서는 chess map과 같이 만들되 링크드 리스트로 만들자
def get_game_over_turn_count(horse_count, game_map, horse_location_and_directions):
    n = len(chess_map)
    current_stacked_horse_map = [[ # [[[] for _ in range(n)] for _ in range(n)] 3차원배열
        [] for _ in range(n)
    ] for _ in range(n)
    ]
    for i in range(horse_count):#현재 쌓여있는 말들의 배열
        r, c, d = horse_location_and_directions[i]
        current_stacked_horse_map[r][c].append(i) #위치만 저장
    
    turn_count = 1 #게임이 종료되는 턴의 번호를 반환해야 하므로
    while turn_count <= 1000: #최대는 1000이므로
        for horse_index in range(horse_count): #말들을 반복
            r, c, d = horse_location_and_directions[horse_index] #말들의 현재 위치와 방향을 알 수 있음
            new_r = r + dr[d] #매 위치가 변경
            new_c = c + dc[d] #매 위치가 변경

            #범위(맵)를 벗어나거나 그 칸이 파랑색인 경우 반대로 한칸 이동
            if not 0 <=new_r < n or not 0<= new_c < n or game_map[new_r][new_c] == 2:
                new_d = get_d_index_when_go_back(d) #새로운 방향

                horse_location_and_directions[horse_index][2] = new_d #방향업데이트
                new_r = r + dr[new_d]
                new_c = c + dc[new_d]
                #뒤집은 곳도 막혀있거나 파랑색이면 이동안함
                if not 0 <=new_r < n or not 0<= new_c < n or game_map[new_r][new_c] == 2:
                    continue

            #[1,2,3] -> 2이동하면 2,3만 이동-자신의 인덱스보다 큰 애들만 데리고 이동
            for i in range(len(current_stacked_horse_map[r][c])):#같이 이동할 말을 알려면 어떻게 쌓여있는지 알아야 함.
                current_stacked_horse_index = current_stacked_horse_map[r][c][i] #쌓여져 있는 말의 인덱스 번호
                if horse_index == current_stacked_horse_index: #현재 이동하고 있는애 horse_index
                    moving_horse_index_array = current_stacked_horse_map[r][c][i:]
                    current_stacked_horse_map[r][c] = current_stacked_horse_map[r][c][:i] #이동 안하고 남아있을 애들
                    break #현재 이동할 애들이 무엇인지 알아내기 위해서
            if game_map[new_r][new_c] == 1: #빨간색인 경우 이동할때 쌓여있는 순서를 반대로 바꿈
                moving_horse_index_array = reversed(moving_horse_index_array)

            #새롭게 이동할 current_stacked_horse_map에 이동한 애들 쌓아주기
            for moving_horse_index in moving_horse_index_array:
                current_stacked_horse_map[new_r][new_c].append(moving_horse_index) #쌓아줌
                horse_location_and_directions[moving_horse_index][0], horse_location_and_directions[moving_horse_index][1] = new_r, new_c #말이 이동했으므로 방금 옮긴 애의 행열값을 업데이트해야함
            #끝나는 조건- 턴이 진행되는 중 말이 4개 이상 쌓이는 순간 게임이 종료된다.
            if len(current_stacked_horse_map[new_r][new_c]) >= 4:
                return turn_count
        turn_count += 1

    return -1


print(get_game_over_turn_count(k, chess_map, start_horse_location_and_directions))  # 2가 반환 되어야합니다