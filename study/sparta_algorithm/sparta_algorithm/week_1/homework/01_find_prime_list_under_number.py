input = 20
#내가 한 것
#소수는 자기 자신과 1 이외에는 아무것도 나눌 수 없다.
# def find_prime_list_under_number(number):
#     numbers = []
#     count = 0
#     for num in range(1, number+1):
#         for i in range(2, num+1):
#             if num % i == 0:
#                 count += 1
#         if count == 1:
#             numbers.append(num)
#         count = 0
#
#     return numbers

#1.효율X
# def find_prime_list_under_number(number):
#     prime_list = []
#     for n in range(2, number+1):
#         for i in range(2, n):
#             if n % i == 0:
#                 break
#         else:
#             prime_list.append(n)
#     return prime_list

#2.효율X
# def find_prime_list_under_number(number):
#     prime_list = []
#     for n in range(2, number+1):
#         # n = 10
#         # i = 2, 3, 5, 7,...
#         # 2 -> X / 3 -> X / 6 ->X
#         #소수로만 나눠서 나누어 떨어지는지 여부 확인하는 것임.
#         for i in prime_list: #이미 이 안에 앞의 소수들이 들어있기 때문
#             # i 의 범위 : 2부터 n - 1 까지의 소수
#             if n % i == 0:
#                 break
#         else:
#             prime_list.append(n)
#     return prime_list

#3.효율최상
#소수는 자기 자신과 1 외에는 아무것도 나눌 수 없다.
#주어진 자연수 N이 소수이기 위한 필요충분조건은
#N이 N의 제곱근보다 크지 않은 어떤 소수로도 나눠지지 않는다.
#수가 수를 나누면 몫이 발생하는데 몫과 나누는 수 둘중 하나는 반드시
#N의 제곱근 이하이다.
def find_prime_list_under_number(number):
    prime_list = []
    for n in range(2, number+1):
        # n = 10
        # i = 2, 3, 5, 7,...
        # 2 -> X / 3 -> X / 6 ->X
        #소수로만 나눠서 나누어 떨어지는지 여부 확인하는 것임.
        for i in prime_list: #이미 이 안에 앞의 소수들이 들어있기 때문
            # i 의 범위 : 2부터 n - 1 까지의 소수
            if n % i == 0 and i * i <= n:
                break
        else:
            prime_list.append(n)
    return prime_list
#알고리즘을 개선하기 위해서는 수학적 특징, 혹은 특정 숫자의 특징이나 개념들을 떠올리면
#풀기 더 쉽다.
result = find_prime_list_under_number(input)
print(result)