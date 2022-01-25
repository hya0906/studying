input = 100
#재귀함수로 100을 하려면 오류가 난다. 너무 많이 돌아서

# memo 라는 변수에 Fibo(1)과 Fibo(2) 값을 저장해놨습니다!
memo = {
    1: 1,
    2: 1
}

#1.만약 메모에 있으면 그 값을 바로 반환하고
#2.없으면 아까 수식대로 구한다.
#3.그리고 그 값을 다시 메모에 기록한다.

#1.반복되는 부분문제가 있으면
#2.메모이제이션을 해야겠다고 생각
#->다이나믹프로그래밍을 하는 방법

#여기서 fibo(100) -> fibo(99) -> fibo(98) ->....으로 내려감 = Top Down방식
#반대로 fibo(1) -> fibo(2) -> ... 으로 올라감 = Bottom Up 방식
def fibo_dynamic_programming(n, fibo_memo):
    if n in fibo_memo:
        return fibo_memo[n]

    nth_fibo = fibo_dynamic_programming(n-1,fibo_memo)+fibo_dynamic_programming(n-2,fibo_memo)
    fibo_memo[n] = nth_fibo
    return nth_fibo


print(fibo_dynamic_programming(input, memo))