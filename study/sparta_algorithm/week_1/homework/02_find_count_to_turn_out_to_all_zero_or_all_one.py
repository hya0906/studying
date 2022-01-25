input = "011110"

#규칙성
#모두 0으로 만드는 방법에서 최소로 뒤집는 숫자
#0->1로 문자열이 전환되는 순간 count_to_all_zero +=1

#모두 1으로 만드는 방법에서 최소로 뒤집는 숫자
#1->0로 문자열이 전환되는 순간 count_to_all_zero +=0

#1)뒤집어질때 즉 0에서 1혹은 1 에서 0으로 바뀔때
#2)첫번째 원소가 0인지 1인지에 따라서 숫자를 추가해야 한다.

def find_count_to_turn_out_to_all_zero_or_all_one(string):
    count_to_all_zero = 0
    count_to_all_one = 0

    if string[0] == '0':
        count_to_all_one += 1
    elif string[0] == '1':
        count_to_all_zero += 1

    for i in range(len(string) - 1): #앞뒤를 봐야하기 때문에 i를 봄
        if string[i] != string[i + 1]: #1->0, 0->1로 바뀐것
            if string[i + 1] == '0':
                count_to_all_one += 1 #1->0이므로 앞의 숫자를 전부 0으로 바꾸어야 한다.
            if string[i + 1] == '1':
                count_to_all_zero += 1
    print(count_to_all_one, count_to_all_zero)

    return min(count_to_all_one, count_to_all_zero)

result = find_count_to_turn_out_to_all_zero_or_all_one(input)
print(result)