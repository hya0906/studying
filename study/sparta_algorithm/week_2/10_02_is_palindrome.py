input = "소주만병만주소"

#재귀함수-문제가 축소되는 특징이 보여야 한다.
#무조건 탈출조건 써야 함.
def is_palindrome(string):
    if len(string) <= 1:
        return True
    if string[0] != string[-1]:
        return False
    return is_palindrome(string[1:-1])


print(is_palindrome(input))