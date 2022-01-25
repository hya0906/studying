# 위의 그래프를 예시로 삼아서 인접 리스트 방식으로 표현했습니다!
graph = {
    1: [2, 3, 4],
    2: [1, 5],
    3: [1, 6, 7],
    4: [1, 8],
    5: [2, 9],
    6: [3, 10],
    7: [3],
    8: [4],
    9: [5],
    10: [6]
}

#1.시작노드를 큐에 넣는다
#2.현재 큐의 노드를 빼서 visited에 추가한다.
#3.현재 방문한 노드와 인접한 노드 중 방문하지 않은 노드를 큐에 추가한다.

def bfs_queue(adj_graph, start_node):
    queue = [start_node]
    visited = []
    while queue:
        current_node = queue.pop(0) #맨 앞부분을 뽑아야 하기때문
        visited.append(current_node)
        for adjacent_node in adj_graph[current_node]: #인접 그래프에서 인접한 노드만 봄
            if adjacent_node not in visited: #인접한 노드가 방문하지 않았으면
                queue.append(adjacent_node)  #큐에 추가
    return visited


print(bfs_queue(graph, 1))  # 1 이 시작노드입니다!
# [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] 이 출력되어야 합니다!