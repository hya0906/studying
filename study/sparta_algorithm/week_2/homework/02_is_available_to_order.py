shop_menus = ["만두", "떡볶이", "오뎅", "사이다", "콜라"]
shop_orders = ["오뎅", "콜라", "만두"]

#2번 = 항상 이분탐색이 좋은 것은 아님
#정렬을 꼭 할 필요 없음 O(N) + O(M) = O(N+M) - 간단해짐
def is_available_to_order(menus, orders):
    menus_set = set(menus) #O(N)
    for order in orders: #m
        if order not in menus_set: #O(1)
            return False

#1번
#총 O(N * logN) O(M*logN)
#O((M+N) * logN)
# def is_available_to_order(menus, orders):
#     menus.sort()  # menus 정렬! 이분탐색은 정렬필요 이분탐색은 O(logN)
#     #정렬의 시간복잡도는 배열의 길이를 n이라고 했을때 O(N * logN)이다
#     for order in orders: #하나씩 뽑아서 이분탐색 돌려봄 - O(M * logN)
#         if not is_existing_target_number_binary(order, shop_menus):#menus->shop_menus로 바꿈 이거 부를때마다 O(logN)
#             return False
#     return True
# 
# def is_existing_target_number_binary(target, array): #그전에 했던 것 복사함
#     current_min = 0
#     current_max = len(array) - 1
#     current_guess = (current_min + current_max) // 2
# 
#     while current_min <= current_max:
#         if array[current_guess] == target:
#             return True
#         elif array[current_guess] < target:
#             current_min = current_guess + 1
#         else:
#             current_max = current_guess - 1
#         current_guess = (current_min + current_max) // 2
# 
#     return False

result = is_available_to_order(shop_menus, shop_orders)
print(result)