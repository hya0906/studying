input = "abcabcabcabcdededededede"
#압축할 문자열중 가장 짧은 것을 반환하는 것 -> 모든 경우를 다 봐서 최소값을 반환해야함
#문자열을 보게 되면 반 이상(의 구간)을 넘어가면 압축이 안됨 n//2부터는 자를 필요가 없음

#내가 문제를 잘 이해하고 구현하는 방법을 생각하는게 알고리즘 풀이의 핵심이자 중요함!!!
def string_compression(string):
    #1부터 n//2까지 쪼개기
    n = len(string)
    compression_length_array = []
    for split_size in range(1, n//2 + 1):
        #splited = []
        ##예- split_size=2이면 0 2 4 6 8 10.. n 이 나옴
        #for i in range(0, n, split_size):
        #    splited.append(string[i:i+split_size]) #2이면 0:2, 2:4, ..이므로 2개씩 쪼갤 수 있음
        splited = [ #위의 주석 4개 한 것과 같은 기능함. 배열을 똑같이 이용할 수 있다.
            string[i:i + split_size] for i in range(0, n, split_size)
        ]
        compressed = "" #압축해서 표현한 문자를 다시 변수에 저장해서 공통으로 쓰기 위해 추가
        #splited 1번째와 2번째비교 ->맞으면 2번째와 3번째 비교...->갯수 저장필요
        count = 1 #이미 자기 자신은 나와있으므로
        print(splited)
        for j in range(1, len(splited)): #이전값과 지금값을 비교하기 위해 1부터 시작
            prev, cur = splited[j - 1], splited[j] #이전, 지금 값 비교하기 위해
            if prev == cur: #똑같다
                count += 1
            else:           #다르다
                if count > 1: #압축 잘 됨 압축할 필요 있음
                                #       2       abc
                    compressed += (str(count) + prev)
                else: #압축할 필요없이 그냥 추가
                           #      abc
                    compressed += prev
                count = 1 #반복이 끝나면 count를 1로 초기화
        #반복문이 끝나고 나서 꼬다리 처리를 해줘야 한다. 맨 마지막에 남은것도 앞과 같은지 확인해야함
        #마지막에 끝난 카운트를 봐줘야한다.
        if count > 1:
            compressed += (str(count) + splited[-1]) #여기서는 prev가 없으므로 splited[-1]
        else:
            compressed += splited[-1]
        compression_length_array.append(len(compressed))
        print(compressed)
    return min(compression_length_array)


print(string_compression(input))  # 14 가 출력되어야 합니다!