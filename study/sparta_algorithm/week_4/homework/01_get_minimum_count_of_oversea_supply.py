import heapq
#최소한의 횟수로 부족한 밀가루를 가져온다.
ramen_stock = 4 #현재 재고량
supply_dates = [4, 10, 15] #공급해주는 날짜
supply_supplies = [20, 5, 10] #공급량
supply_recover_k = 30 #원래 공장으로부터 공급받을 수 있는 시점(k이후 부터는 공급받지 않아도 된다는 날짜의미)

#제일 많은 supplies중에서 가장 많은 것들을 가져오면 된다는 생각- 밀가루 양을 supplies 최고로 높은 순서로 내림차순으로 정렬한 다음에 큰값을 넣으면 되겠다는 생각(X)
#-> dates 0 1 2 날짜는 계속 증가
#   stock 2 1 0  -> 무조건 공장은 가동되도록 해야함 마이너스 안됨.
#현재 재고가 바닥나는 시점 이전까지 받을 수 있는 밀가루 중 제일 많은 밀가루를 받는 것이 목표
#1.현재 재고의 상태에 따라 최고값을 받아야 한다.-> 동적으로 변경되는 상황(현재 재고가 바닥나는지 아닌지 여부에 따라 동적으로 그 최고값을 변경해야 한다는 의미)
#2.제일 많은 값, 제일 큰 값을 뽑아야 한다.

#이 두가지를 생각했을때
#1.데이터를 넣을때마다 최댓값을 동적으로 변경시키며
#2.최소/최대값을 바로 꺼낼 수 있는 자료구조를 사용하면 좋겠다. -> Heap!!

#우리의 목표는 현재 재고stock이 k보다 많아지면 됨. k일 이후에는 받을 필요가 없으므로
def get_minimum_count_of_overseas_supply(stock, dates, supplies, k): #supplies에서 가장 최소로 공급받을 수 있는 횟수 중요
    answer = 0
    last_added_date_index = 0
    max_heap = []

    while stock <= k:
        #예-stock이 10이라고 한다면 dates는 8 11이 있을것-8일까지는 기다릴 수 있음 10일까지는 안됨
        #무조건 dates<stock이여야 함
        #여기서 date를 일일이 다 조회함-언제까지 버틸 수 있는 것응 알고 supplies중 20을 max heap에 담아서 가장 큰 수를 stock에 추가
        #     아래의 이 조건이 없으면 무한대로 값이 나올때까지 돌음-에러
        while last_added_date_index < len(dates) and dates[last_added_date_index] <= stock:
            # 4일까지 버틸 수 있으니까 0번째 인덱싱 20이라는 supplies를 max heap에 넣겠다란 의미
            #그러면 maxheap에는 20이 담겨져 있음
            heapq.heappush(max_heap, -supplies[last_added_date_index]) #supplies에 있는 값을 넣어서 stock에다가 추가-heap사용
            #힙의 성질로 가장 큰 값을 빼서 스톡에 추가할 수 있기 때문
            last_added_date_index += 1 #가장 마지막에 더했던 날짜의 index. 추가함으로써 다음에 넣을 공급량의 배열들을 순회한다.
            #넣고 싶은 공급량을 우선순위에 맞춰서 힙에 정렬해서 넣어둠
        answer += 1
        #얘네들이 maxheap을 구현하고 싶지만 minheap만 지원하므로 마이너스로 이런 방법을 쓴다.
        heappop = heapq.heappop(max_heap) #최상값 빼내기- 실제로 사용할때는 -를 붙여서 사용해야하므로 실제 내용에는 -20이지만 사용할 내용으로는 20이 들어가있다.
        stock += -heappop #재고에 계속 넣음
    return answer


print(get_minimum_count_of_overseas_supply(ramen_stock, supply_dates, supply_supplies, supply_recover_k))
print("정답 = 2 / 현재 풀이 값 = ", get_minimum_count_of_overseas_supply(4, [4, 10, 15], [20, 5, 10], 30))
print("정답 = 4 / 현재 풀이 값 = ", get_minimum_count_of_overseas_supply(4, [4, 10, 15, 20], [20, 5, 10, 5], 40))
print("정답 = 1 / 현재 풀이 값 = ", get_minimum_count_of_overseas_supply(2, [1, 10], [10, 100], 11))
#데이터를 넣을때마다 최대값을 동적으로 변경시키면서 최솟값, 최댓값을 바로 꺼낼 수 있는 구조가 필요하다->heap사용!!