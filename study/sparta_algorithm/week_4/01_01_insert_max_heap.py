class MaxHeap:
    def __init__(self):
        self.items = [None] #완전이진트리를 배열로 하기 위해서는 맨처음에 none값을 해야한다.
        # 편리하게 하기 위해

#1.새 노드를 맨 끝에 추가한다.
#2.지금 넣은 새 노드와 부모를 비교한다. 만약 부모보다 크다면 교체한다.
#3.이 과정을 꼭대기까지 반복한다.
    def insert(self, value): #여기서는 배열로 하고 있으니까 노드를 만들지 않고 그냥 value를 append해준다.
        self.items.append(value)
        cur_index = len(self.items) - 1
        while cur_index > 1: #index가 1이라면 멈춰라
            parent_index = cur_index // 2
            if self.items[cur_index] > self.items[parent_index]:
                self.items[cur_index], self.items[parent_index] = self.items[parent_index], self.items[cur_index]
                cur_index = parent_index
            else:
                break
        return


max_heap = MaxHeap()
max_heap.insert(3)
max_heap.insert(4)
max_heap.insert(2)
max_heap.insert(9)
print(max_heap.items)  # [None, 9, 4, 2, 3] 가 출력되어야 합니다!
#      9
#    4   2
#  3