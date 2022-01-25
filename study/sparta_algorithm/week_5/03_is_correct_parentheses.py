#모든 괄호를 뽑아서 올바른 순서대로 배치된 괄호 문자열을 알려주는 프로그램
#개수도 같고 짝도 같은 문자열
#이 문제에는 문제에서 어떻게 하라는 과정이 다 나와있어서(힌트존재) 따라서 구현하기만 하면 됨(간혹 이런 경우가 몇몇 있음)
from collections import deque

balanced_parentheses_string = "()))((()" #균형잡힌 괄호 문자열
#균혐잡힌 괄호 문자열 -> 올바른 괄호 문자열
#올바른 괄호 문자열? 어떻게 알았지? -> 3주차 2번째(비슷한 제목)문제와 비슷

def is_correct_parenthesis(string): #올바른 괄호 문자열인지 확인해주는 함수
    stack = []
    for s in string:
        if s == "(": #열린거면 추가
            stack.append(s)
        elif stack:        #닫힌거면 뺌
            # if len(stack) == 0: #조건문 대신 elif로 해결-없을때 pop을 하면 에러
            #     return "에러"
            stack.pop()
    return len(stack) == 0

def reverse_parenthesis(string):
    reversed_string = ""  # 4-4구현
    for char in string:
        if char == "(":
            reversed_string += ")"
        else:
            reversed_string += "("
    return reversed_string

def separate_to_u_v(string):
    queue = deque(string)
    left, right = 0, 0
    u, v = "", ""
    while queue:  # 하나씩 빼면서 (와 )의 개수가 같다는 걸 확인해야함
        char = queue.popleft()
        u += char  # 하나씩 꺼내면서 u에다가 붙임
        if char == "(":
            left += 1  # (괄호 개수를 한개씩 저장해줌
        else:
            right += 1  # )괄호 개수를 한개씩 저장해줌
        # 균형잡힌 문자열의 조건임
        if left == right:  # 단~~이부분 참고, u가 균형잡힌 괄호 문자열이 되지 않도록 하기 위해서 더이상 쌍이
            break  # 안 생기도록 맨 처음 left와 right가 맞았을때 멈추도록 해야한다.
    #남아있는 queue에 있는 데이터들은 v에다가 다 넣어주면 된다.->모두 다 담는다고 해서 어떻게 균형잡힌 문자열이 될 수 있나?
    # ->이유-애초에 문자열을 균형잡힌 문자열만 넣어준다.(전제조건) -> 그렇기 때문에 v도 반드시 균형이 잡혔을 것이라고 생각할 수 있다.
    v = ''.join(list(queue)) #""를 기준으로 문자열 붙임
    return u, v

#1.입력이 빈 문자열인 경우, 빈 문자열 반환
def change_to_correct_parenthesis(string):#균형잡힌 괄호 문자열을 올바른 괄호문자열로 바꾸는 함수
    if string == "":
        return ""
    # 2.문자열 w를 두 "균형잡힌 괄호 문자열" u,v로 분리할 수 있어야 한다.
    # ->균형잡힌 괄호 문자열-> (와 )의 개수가 같아야 한다.
    # 단, u는 "균형잡힌 괄호 문자열"로 더이상 분리할 수 없어야 하며
    # v는 빈 문자열이 될 수 있습니다.
    u, v = separate_to_u_v(string)

    #3.문자열 u가 "올바른 괄호 문자열"이라면 문자열 v에 대해
    #1단계부터 다시 수행합니다.
    #3-1.수행한 결과 문자열을 u에 이어붙인 뒤 반환합니다.
    #->change_to_correct_parenthesis (재귀적)-1번부터 다시
    if is_correct_parenthesis(u):
        return u + change_to_correct_parenthesis(v)

    #4.문자열 u가 올바른 괄호 문자열이 아니라면 아래 과정을 수행합니다.
    #4-1.빈 문자열에 첫번째 문자로 (를 붙입니다.
    #4-2.문자열 v에 대해 1단계부터 재귀적으로 수행한 결과 문자열을 이어붙입니다.
    #4-3.)를 다시 붙입니다.
    #4-4.u의 첫번째 문자열과 마지막 문자를 제거하고, 나머지 문자열의 괄호 방향을
    #뒤집어서 뒤에 붙입니다.
    else:
                                                       # 4-4 첫번째 문자와 마지막 문자를 제거하라고 했으므로
        return "(" + change_to_correct_parenthesis(v) + ")" + reverse_parenthesis(u[1:-1]) #u가 올바른 괄호 문자열이 아니라면.. #4-1,2,3구현 + 4-4



def get_correct_parentheses(balanced_parentheses_string):
    if is_correct_parenthesis(balanced_parentheses_string): #원래 올바른 문자열이기때문에 고칠 것이 없음
        return balanced_parentheses_string
    else:#문제에 쓰여있는 방식대로 해주면 됨
        return change_to_correct_parenthesis(balanced_parentheses_string)


print(get_correct_parentheses(balanced_parentheses_string))  # "()(())()"가 반환 되어야 합니다!