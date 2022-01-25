#원소의 최대값을 빠르게 추가하고 삭제하는 문제풀이때 사용
class MaxHeap:
    def __init__(self):
        self.items = [None]

    def insert(self, value):
        self.items.append(value)
        cur_index = len(self.items) - 1

        while cur_index > 1:  # cur_index 가 1이 되면 정상을 찍은거라 다른 것과 비교 안하셔도 됩니다!
            parent_index = cur_index // 2
            if self.items[parent_index] < self.items[cur_index]:
                self.items[parent_index], self.items[cur_index] = self.items[cur_index], self.items[parent_index]
                cur_index = parent_index
            else:
                break
#1.루트 노드와 맨 끝에 있는 원소를 교체한다.
#2.맨 뒤에 있는 원소(원래 루트노드)를 삭제한다. 이때 끝에 반환해줘야 하므로 저장해야함
#3.변경된 노드와 자식노드 비교. 두 자식중 자신보다 자식이 더 크면 바꾼다
#4.자식노드 둘보다 부모노드가 더 크거나 가장 바닥에 도달하지 않을때까지 3을 반복한다.
#5.2에서 제거한 원래 루트 노드를 반환한다.
    def delete(self):
        self.items[1], self.items[-1] = self.items[-1], self.items[1] #루트노드와 맨 끝에 있는 노드 바꾸기
        prev_max = self.items.pop() #뽑은 값 저장

        cur_index = 1

        while cur_index <= len(self.items) - 1:
            left_child_index = cur_index * 2
            right_child_index = cur_index * 2 + 1
            max_index = cur_index #왼쪽이 큰지 오른쪽이 큰지 비교
            #cur,max,min중 가장 큰것만 찾으면 됨 그래서 cur_index가 아니라 max_index를 사용함.(아래 if문끝에서)
                #현재 왼쪽에 자식이 있다는 의미
            if left_child_index <= len(self.items) - 1 and self.items[left_child_index] > self.items[max_index]:
                max_index = left_child_index

            if right_child_index <= len(self.items) - 1 and self.items[right_child_index] > self.items[max_index]:
                max_index = right_child_index

            if max_index == cur_index: #현재 노드가 자식들보다 크다는 의미
                break

            self.items[cur_index], self.items[max_index] = self.items[max_index], self.items[cur_index]
            cur_index = max_index
        return prev_max  # 8 을 반환해야 합니다.


max_heap = MaxHeap()
max_heap.insert(8)
max_heap.insert(6)
max_heap.insert(7)
max_heap.insert(2)
max_heap.insert(5)
max_heap.insert(4)
print(max_heap.items)  # [None, 8, 6, 7, 2, 5, 4]
print(max_heap.delete())  # 8 을 반환해야 합니다!
print(max_heap.items)  # [None, 7, 6, 4, 2, 5]