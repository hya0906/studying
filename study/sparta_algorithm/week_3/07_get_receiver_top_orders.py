top_heights = [6, 9, 5, 7, 4]

# <- <- <- <-

#비교 후에는 맨마지막을 없애준다. 이미 완료해서 필요없기때문
#1개가 남으면 비교할 곳이 없으므로 안해줘도 된다.
def get_receiver_top_orders(heights): #O(N^2)
    answer = [0] * len(heights)
    while heights: #O(N)
        height = heights.pop()
        # [6, 9, 5, 7]
        for idx in range(len(heights)-1, 0, -1): #O(N)
            print(answer,idx,heights[idx])
            if heights[idx] > height:
                answer[len(heights)] = idx + 1
                print(answer)
                print("+++++++++")
                break
    return answer


print(get_receiver_top_orders(top_heights))  # [0, 0, 2, 2, 4] 가 반환되어야 한다!