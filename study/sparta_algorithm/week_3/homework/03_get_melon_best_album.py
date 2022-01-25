genres = ["classic", "pop", "classic", "classic", "pop"]
plays = [500, 600, 150, 800, 2500]
#장르순->플레이순
#1. 속한 노래가 많이 재생된 장르를 먼저 수록한다. ->장르별로 노래순으로 정렬
#2. 장르 내에서 많이 재생된 노래를 먼저 수록한다.
#3. 장르 내에서 재생횟수가 같다면 고유 번호가 낮은 노래 먼저 수록한다.
#1- 장르별(key)로 재생된 횟수(value)를 저장해야함.->딕셔너리
#2- 장르별로 플레이 수가 몇인지 저장해야함. 인덱스와 함께-> 장르별로 곡의 정보(인덱스, 재생횟수) 배열로 묶어 저장한다.

def get_melon_best_album(genre_array, play_array):
    n = len(genre_array)
    genre_total_play_dict = {}
    genre_index_play_array_dict = {}
    for i in range(n): #장르의 플레이 수를 딕셔너리에 넣음
        genre = genre_array[i]
        play = play_array[i]
        if genre not in genre_total_play_dict: #초기값때문 없으면 대입. 있으면 더함.
            genre_total_play_dict[genre] = play
            genre_index_play_array_dict[genre] = [[i, play]] #배열로 만듬. 장르별로 여러 곡이 쌓일 것이기 때문
        else:
            genre_total_play_dict[genre] += play
            genre_index_play_array_dict[genre].append([i, play])
    #print(genre_total_play_dict)
    #print(genre_index_play_array_dict)
    #딕셔너리 정렬-                                                      뒤에있는 값을 기준으로 정렬, 내림차순으로
    sorted_genre_play_array = sorted(genre_total_play_dict.items(), key=lambda item: item[1], reverse=True)
    #print(sorted_genre_play_array)
    result = []
    for genre, _value in sorted_genre_play_array:
        index_play_array = genre_index_play_array_dict[genre]
        #두번째 index_play도 정렬해야 함
        sorted_by_play_and_index_play_index_array = sorted(index_play_array, key=lambda item: item[1], reverse=True)
        #print(sorted_by_play_and_index_play_index_array)
        for i in range(len(sorted_by_play_and_index_play_index_array)):
            if i > 1: # 장르별로 2개만이므로
                break
            result.append(sorted_by_play_and_index_play_index_array[i][0]) #곡의 인덱스값 반환
    return result


print(get_melon_best_album(genres, plays))