def solution(graph, node1, node2):
    visited = []
    path = []
    def dfs(node):
        print(f"current node is {node}")
        visited.append(node)
        path.append(node)
        if node==node2: # 成功找到了
            return True
        for neighbor_node in graph[node]:
            if neighbor_node not in visited:
                print(f'current neighbor node is {neighbor_node}')
                print(f'current path is {path}')
                # 必须要接受返回值
                res = dfs(neighbor_node)
                if res: # 表示在neighbor_node的邻居节点中找到了node2
                    # 如果不接收返回值的话，会导致父节点不知道信号，由此继续执行
                    return path[:]
                path.pop()
        return False # 没有成功找到
    
    dfs(node1)
    return path

graph = [
  [1],
  [0, 2, 5, 4],
  [1, 4, 5],
  [],
  [5, 2, 1],
  [1, 2, 4]
]
node1 = 0
node2 = 4
print(solution(graph, node1, node2))
