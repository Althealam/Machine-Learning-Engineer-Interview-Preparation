def solution(graph):
    directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    m, n = len(graph), len(graph[0])
    
    def dfs(graph, i, j, k, visited):
        graph[i][j] = k
        visited[i][j] = True
        for dx, dy in directions:
            next_i, next_j = i+dx, j+dy
            if 0<=next_i<m and 0<=next_j<n and not visited[next_i][next_j] and graph[next_i][next_j]==1:
                dfs(graph, next_i, next_j, k, visited)
    
    visited1 = [[0]*n for _ in range(m)]
    visited2 = [[0]*n for _ in range(m)]
    visited3 = [[0]*n for _ in range(m)]
    visited4 = [[0]*n for _ in range(m)]
    for i in range(m): # let the points which are on the very left/right/top/bottom be 2
        if graph[i][0]==1:
            dfs(graph, i, 0, 2, visited1)
        if graph[i][n-1]==1:
            dfs(graph, i, n-1, 2, visited2)
    for j in range(n):
        if graph[0][j]==1:
            dfs(graph, 0, j, 2, visited3)
        if graph[m-1][j]==1:
            dfs(graph, m-1, j, 2, visited4)
    
    visited = [[False]*n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            if graph[i][j]==1:
                dfs(graph, i, j, 0, visited)

    for i in range(m):
        for j in range(n):
            if graph[i][j]==2:
                graph[i][j] = 1
    return graph



graph = [
    [1, 0, 0, 0, 0, 0], 
    [0, 1, 0, 1, 1, 1],
    [0, 0, 1, 0, 1, 0],
    [1, 1, 0, 0, 1, 0], 
    [1, 0, 1, 1, 0, 0], 
    [1, 0, 0, 0, 0, 1]
]

sample_output = [
    [1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 1, 0],
    [1, 1, 0, 0, 1, 0],
    [1, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1]
]

res = solution(graph)
print(res)