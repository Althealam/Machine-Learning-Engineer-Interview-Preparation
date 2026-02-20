def solution(S):
    res = []
    backtracking(res, [], 0, S)
    return res

def backtracking(res, path, startIndex, S):
    res.append(path[:])
    for i in range(startIndex, len(S)):
        path.append(S[i])
        backtracking(res, path, i+1, S)
        path.pop()

S = ['x', 'y', 'z']
print(solution(S))