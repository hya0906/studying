import itertools, sys

n = 5
m = 3

city_map = [ #현재는 치킨 집이 3개 밖에 없기 때문에 튜플이 1개만 나옴. [([1,2], [2,2], [4,4])]
    [0, 0, 1, 0, 0], #2가 더 있으면 조합이 여러개 나옴
    [0, 0, 2, 0, 1],
    [0, 1, 2, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 0, 2],
]
#여기서는 입력값의 범위가 상당히 작으므로 for문을 여기처럼 여러개 사용해도 된다. 무식하게 풀어도 된다.

#여러개 중에서 m개를 고른 뒤 모든 치킨 거리의 합이 가장 작게 되는 경우
#->여러개 중에서 특정 개수를 뽑는 경우의 수
#->모든 경우의 수를 다 구해야 함. =>조합 사용!!
def get_min_city_chicken_distance(n, m, city_map):
    chicken_location_list = [] #치킨집 위치
    home_location_list = [] #집 위치
    for i in range(n): #city맵 행과 열
        for j in range(n):
            if city_map[i][j] == 1:
                home_location_list.append([i, j])
            if city_map[i][j] == 2:
                chicken_location_list.append([i, j])

    # 치킨집 중에 M개 고르기(조합)
    chicken_location_m_combinations = list(itertools.combinations(chicken_location_list, m))
    min_distance_of_m_combinations = sys.maxsize #최소 도시 치킨 거리
    for chicken_location_m_combination in chicken_location_m_combinations: #도시 치킨거리가 얼마나 될지
        city_chicken_distance = 0 #도시 치킨거리는 각 집들의 치킨거리의 합이므로
        for home_r, home_c in home_location_list:
            min_home_chicken_distance = sys.maxsize #최소값 치킨거리
            for chicken_location in chicken_location_m_combination: #각 집의 치킨 거리를 구함
                min_home_chicken_distance = min(
                    min_home_chicken_distance,
                    abs(home_r - chicken_location[0]) + abs(home_c - chicken_location[1])
                )   #집의 위치와 치킨집 위치의 절대값 비교
            city_chicken_distance += min_home_chicken_distance
        min_distance_of_m_combinations = min(min_distance_of_m_combinations, city_chicken_distance)
    return min_distance_of_m_combinations


# 출력
print(get_min_city_chicken_distance(n, m, city_map))  # 5 가 반환되어야 합니다!