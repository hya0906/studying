input = "(())()"
#1.( 괄호 열림       stack = ["("]
#2. ( ( 괄호 또 열림 stack = ["(", ")"]
#3. )괄호 닫힘. 짝지어진건 사라짐. 그러면 아까 열린 것 중에 현재 열린 괄호는 (  # stack = ["("]
#4. )괄호 닫힘. 현재 열린 괄호 없음    stack = []
#5. ( 다시 괄호 열림                stack = ["("]
#6. )괄호 닫힘 -> 마지막에 아무것도 남지 않았으므로 올바른 괄호쌍이라는 것을 알게 됨. stack = []

#((()
#1.( 괄호가 열림    stack = ["("]
#2.(( 괄호가 열림   stack = ["(", "("]
#3.((( 괄호가 열림  stack = ["(","(","("]
#4.)괄호가 닫힘. 현재 열린 괄호 (( -> 그런데 끝남 stack = ["(","("] -> 남아있음
#->괄호쌍이 올바르지 않다.

#1.직전에 열린 괄호가 있는지 본다. 열려있으면 닫으면 됨
#2.열린 괄호를 계속 저장해 놔야 한다.
#->바로 직전에 조회한 괄호를 저장해야 한다, -> 즉 stack 사용

def is_correct_parenthesis(string):
    stack = []

    for i in range(len(string)):
        if string[i] == "(":
            stack.append(i)  # 여기 아무런 값이 들어가도 상관없습니다! ( 가 들어가있는지 여부만 저장해둔 거니까요
        elif string[i] == ")":
            if len(stack) == 0: #스택에 아무것도 없는데 닫는 괄호가 나오는 경우
                return False
            stack.pop()

    if len(stack) == 0:
        return True
    else:
        return False


print(is_correct_parenthesis(input))  # True 를 반환해야 합니다!