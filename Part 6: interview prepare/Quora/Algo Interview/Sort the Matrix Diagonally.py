# (0, 0), (1, 1), (2, 2) is a combination of path 1 ==> i-j=0
# (0, 1), (1, 2), (2, 3) is a combination of path 2 ==> i-j=-1
# (0, 2), (1, 3) is a combination of path 3 ==> i-j==-2
# (1, 0), (2, 1) is a combination of path 4 ==> i-j==1
# (2, 0) is a combination of path 5 ==> i-j=2
# (0, 3) is a combination of path 6 ==> i-j=-3

from collections import defaultdict
class Solution:
    def diagonalSort(self, mat: List[List[int]]) -> List[List[int]]:
        group = defaultdict(list)
        for i in range(len(mat)):
            for j in range(len(mat[0])):
                group[i-j].append(mat[i][j])
        
        for k in group:
            group[k].sort(reverse=True)

        for i in range(len(mat)):
            for j in range(len(mat[0])):
                mat[i][j] = group[i-j].pop()
        return mat


