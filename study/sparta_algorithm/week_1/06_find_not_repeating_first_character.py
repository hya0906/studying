input = "abadabac"
#내가 한 것
# def find_not_repeating_character(string):
#     for i in string:
#         if string.count(i) == 1:
#             return i

def find_not_repeating_character(string):
    alphabet_occurrence_array = [0] * 26
    for i in string:
        if i.isalpha():
            array_index = ord(i) - ord('a')
            alphabet_occurrence_array[array_index] += 1
        else:
            continue

    not_repeating_character_array = []
    for index in range(len(alphabet_occurrence_array)):
        alphabet_occurrence = alphabet_occurrence_array[index]
        if alphabet_occurrence == 1:
            not_repeating_character_array.append(chr(index+ord('a')))
        print(not_repeating_character_array)#기존 문자열의 순서를 보장해 주지 않음.

    for char in string: #꼭 필요
        if char in not_repeating_character_array:
            return char
    return i


result = find_not_repeating_character(input)
print(result)