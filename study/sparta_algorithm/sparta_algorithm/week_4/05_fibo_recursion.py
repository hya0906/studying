input = 20

#fibo
#->Fibo(4)를 구하기 위해서는
#   ->fibo(3)을 구하고 1을 더함
#   ->fibo(3)을 구하기 위해서는
#       ->fibo(2) 와 fibo(1)을 더해주면 된다.##
#       ->연산량 2번
#->Fibo(3)을 구하기 위해서는
#   ->fibo(2)와 fibo(1)을 더함 ## 위에서 했던 작업을 똑같이 다시 반복
#->이렇게 반복하지 않고 그 전에 했던 것들을 기억하기 위해서 동적계획법이 필요하다 dynamic programming



#input을 100으로 하면 계속 돌아가서 값 안나옴
#Fibo(N) = Fibo(N-1) + Fibo(N-2)
#fibo(1) = fibo(2) = 1
def fibo_recursion(n):
    if n==1 or n == 2:
        return 1
    return fibo_recursion(n-1)+fibo_recursion(n-2)


print(fibo_recursion(input))  # 6765