#numbers = [2, 3, 1] #간소화해서
numbers = [1,1,1,1,1]
#target_number = 0 #간소화해서 풀어보자
target_number = 3
result_count = 0  # target 을 달성할 수 있는 모든 방법의 수를 담기 위한 변수

#규칙성을 알기 어려워서 모든 경우의 수를 다 해봐야 한다.
#1. 2+3+1 = 6 +++ 반복됨-앞에 있는 두개부호를 고정시키면 맨뒤의 것을 +-로 2가지가 추가적으로 생긴다.
#2. 2+3-1 = 4 ++-
#3. 2-3+1 = 0 타겟 +-+
#4. 2-3-1 = -2 +--
#5. -2+3+1 = 2
#6. -2+3-1 = 0 타겟
#7.-2-3+1 = -4
#8. -2-3-1 = -6

#N의 길이의 배열에서 더하거나 뺀 모든 경우의 수는
#N-1의 길이의 배열에서 마지막 원소를 더하거나 뺀 경우의 수를
#추가하면 된다.
#예- [2,3]
#1. +2 +3 -> 1을 더할거냐 뺄거냐에서 2가지 경우 추가
#2. +2 -3
#3. -2 +3
#4. -2 -3
result = [] #모든 변수를 담기위한           들어있는숫자, 어디에 있는지               =result
def get_all_ways_to_by_doing_plus_or_minus(array, current_index, current_sum, all_ways): #모든 경우의 수를 만들어내는 함수
    if current_index == len(numbers):
        all_ways.append(current_sum)
        return

    get_all_ways_to_by_doing_plus_or_minus(array, current_index + 1, current_sum + numbers[current_index], all_ways)
    get_all_ways_to_by_doing_plus_or_minus(array, current_index + 1, current_sum - numbers[current_index], all_ways)

print(get_all_ways_to_by_doing_plus_or_minus(numbers, 0, 0, result))
print(result)

result_count = 0
def get_count_of_ways_to_target_by_doing_plus_or_minus(array, target, current_index, current_sum): #타겟만 만들어내는 함수
    if current_index == len(array):  # 탈출조건!
        if current_sum == target: #값이 동일해야만 1증가
            global result_count #외부의 변수를 변경해주고자 할때,쓰고자 할때 내부에서 사용해야함
            result_count += 1  # 마지막 다다랐을 때 합계를 추가해주면 됩니다.
        return
    get_count_of_ways_to_target_by_doing_plus_or_minus(array, target, current_index + 1,
                                                       current_sum + array[current_index])
    get_count_of_ways_to_target_by_doing_plus_or_minus(array, target, current_index + 1,
                                                       current_sum - array[current_index])


get_count_of_ways_to_target_by_doing_plus_or_minus(numbers, target_number, 0, 0)
# current_index 와 current_sum 에 0, 0을 넣은 이유는 시작하는 총액이 0, 시작 인덱스도 0이니까 그렇습니다!
print(result_count)  # 2가 반환됩니다!

#문제가 축소되는 과정에서는 재귀함수로 해결할 수 있다.
#내가 헷갈리면 모든 경우의 수를 해보고 트라이해보자는 생각을 하자