seat_count = 9
vip_seat_array = [4, 7]
#[1,2,3,4,5,6,7,8,9]-> 4와 7이 고정되어있을때 만들 수 있는 여러가지의 가능한 개수를 알아보는 문제
#좌석[1,2] 2개
#[1,2][2,1]

#좌석[1,2,3] 3개
#[1,2,3],[2,1,3],[1,3,2] ->최대 한자리만 옮길 수 있어서 [3,2,1]이런건 안됨. 두자리씩 옮겼으므로

#좌석[1,2,3,4] 5개
#[1,2,3,4] [1,2,4,3] [1,3,2,4] [2,1,3,4] [2,1,4,3]

#좌석[1,2,3,4,5] 8개
#[1,2,3,4,5] [1,2,3,5,4] [2,1,3,4,5] [2,1,3,5,4] [1,2,4,3,5]
#[2,1,4,3,5] [2,1,3,4,5] [1,3,2,4,5]
#->피보나치 수열!
#F(1) = 1 / F(2) =1
#좌석(2) = 2 / 좌석(3) = 3

# 예전에 만들었던 fibo_dynamic_programming 에서 가져오면 됩니다!
memo = {
    1: 1,  # 이 문제에서는 Fibo(1) = 1, Fibo(2) = 2 로 시작합니다!
    2: 2
}
#좌석이 i개 있다.
#경우의 수 2가지- 그대로 앉는 경우, 바꿔서 앉는 경우
#1.              i 번째 좌석에 i번째 티켓을 가진 사람이 그대로 앉는 경우
#1 2 3 4 5 6 ... i ->★ 1번 경우의 수!!! - i 가 고정이 되어있으면 i-1의 좌석들을 맘껏 배치할 수 있다.
#2. i번째 티켓을 가진 사람이 1번째에 앉을 경우 - 불가능 ... i-1번째 앉을때만 가능

#            i  i-1 -> i번째 티켓가진 애가 i-1에 앉았을 경우 무조건 i-1티켓 가진 애는 i번째에 앉아야 한다.
#1,2,3,4,5...i-1 i인경우       -> 왜냐하면 i-1이 i-2번째 앉을 경우 쪼로록 자신의 앞자리에
#                                앉게 되면 1번은 앉을 자리가 없으므로-> i 번째는 공석이 되어버린다.
#★ 2번째 경우의 수!!!! - i 와 i-1번째가 고정해서 반대로 앉으면 i-2번째의 좌석들은 마음대로 배치 가능하다.


#F(N) = N명의 사람들을 좌석에 배치하는 방법
#     = N-1명의 사람들을 좌석에 배치하는 방법 + N-2명의 사람들을 좌석에 배치하는 방법
#     = F(N-1) + F(N-2)
def fibo_dynamic_programming(n, fibo_memo):
    if n in fibo_memo:
        return fibo_memo[n]

    nth_fibo = fibo_dynamic_programming(n - 1, fibo_memo) + fibo_dynamic_programming(n - 2, fibo_memo)
    fibo_memo[n] = nth_fibo
    return nth_fibo

#1,2,3 F(3) = 3
#5,6   F(2) = 2
#8,9   F(2) = 2  -> 각각 독립적으로 시행됨
#3*3*2 = 12 모든 가지의 수
def get_all_ways_of_theater_seat(total_count, fixed_seat_array):
    all_ways = 1 #곱연산을 해야하기 때문에 1로 지정/또는 아무 자리도 안 움직이면 1로 시작
    current_index = 0
    #[(1,2,3)4(5,6)7(8,9)]
    for fixed_seat in fixed_seat_array: # 4,7
        fixed_seat_index = fixed_seat - 1 #번호로 만들어주기 위해서
                                                 #사이에 있는 좌석의 개수를 구하기 위해 -> 앞의 1,2,3의 n의 개수
        count_of_ways = fibo_dynamic_programming(fixed_seat_index - current_index, memo)
        #각 사이의 좌석마다 나올 수 있는 경우의 수 를 곱연산 함
        all_ways *= count_of_ways
        current_index = fixed_seat_index + 1 #이미 1,2,3을 해줘서 4다음의 인덱스를 보라고 하기 위해 +1함
    #앞에서는 (1,2,3), (5,6)까지만 하고 반복문이 끝나게 된다. 하지만 뒤에 남아있을 생략된 애들도 있다.
    #그래서 for문에 끝나더라도 그것에 대한 계산을 해줘야 한다.
    count_of_ways = fibo_dynamic_programming(total_count - current_index, memo) #9-7 맨 마지막이 7이었으므로
    all_ways *= count_of_ways
    return all_ways

# 12가 출력되어야 합니다!
print(get_all_ways_of_theater_seat(seat_count, vip_seat_array))