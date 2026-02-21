# 1. 获取斜前缀和
# 假设一个菱形的中心在(r, c)，半径为k，那么四个顶点为(r-k, c), (r+k, c), (r, c+k), (r, c-k)
# s1[i][j] = s1[i-1][j-1]+grid[i][j]
# s2[i][j] = s2[i-1][j+1]+grid[i][j]
# 2. 获取菱形的边和

from collections import Counter
class Solution:
    def getBiggestThree(self, grid: list[list[int]]) -> list[int]:
        m, n = len(grid), len(grid[0])
        prefix_sum1 = [[0]*n for _ in range(m)] # start from the left side
        prefix_sum2 = [[0]*n for _ in range(m)] # start from the right side

        for i in range(m):
            for j in range(n):
                prefix_sum1[i][j] = grid[i][j]+(prefix_sum1[i-1][j-1] if i>0 and j>0 else 0)
                prefix_sum2[i][j] = grid[i][j]+(prefix_sum2[i-1][j+1] if i>0 and j<n-1 else 0)

        # # initialize prefix_sum1
        # for i in range(m):
        #     prefix_sum1[i][0]=grid[i][0]
        # for j in range(n):
        #     prefix_sum1[0][j]=grid[0][j]
        
        # # initialize prefix_sum2
        # for i in range(n):
        #     prefix_sum2[i][n-1]=grid[i][n-1]
        # for j in range(m-1, -1, -1):
        #     prefix_sum2[0][j]=grid[0][j]

        # # get the prefix sum for prefix_sum1 and prefix_sum2
        # for i in range(1, m):
        #     for j in range(1, n):
        #         prefix_sum1[i][j]=prefix_sum1[i-1][j-1]+grid[i][j]
        
        # for i in range(1, m-1):
        #     for j in range(n-1):
        #         prefix_sum2[i][j]=prefix_sum2[i-1][j+1]+grid[i][j]

        cnt = Counter()
        for i in range(m): # 枚举中心点横坐标
            for j in range(n): # 枚举中心点纵坐标
                cnt[grid[i][j]]+=1 # 半径为0的情况
                for k in range(1, m): # 枚举半径
                    # for central point (i, j), its neighbor nodes are:
                    # (i-k, j), (i+k, j), (i, j-k), (i, j+k)
                    if i-k<0 or j-k<0 or i+k>=m or j+k>=n:
                        break
                    top = (i-k, j)
                    left = (i, j-k)
                    bottom = (i+k, j)
                    right = (i, j+k)

                    # 利用前缀和计算四条边的和
                    # 1. top->right
                    s1 = prefix_sum1[right[0]][right[1]]-prefix_sum1[top[0]][top[1]]+grid[top[0]][top[1]]
                    # 2. left->bottom
                    s2 = prefix_sum1[bottom[0]][bottom[1]]-prefix_sum1[left[0]][left[1]]+grid[left[0]][left[1]]
                    # 3. top->left
                    s3 = prefix_sum2[left[0]][left[1]]-prefix_sum2[top[0]][top[1]]+grid[top[0]][top[1]]
                    # 4. right->bottom
                    s4 = prefix_sum2[bottom[0]][bottom[1]]-prefix_sum2[right[0]][right[1]]+grid[right[0]][right[1]]

                    total = s1+s2+s3+s4-(grid[top[0]][top[1]]+grid[left[0]][left[1]]+grid[bottom[0]][bottom[1]]+grid[right[0]][right[1]])
                    cnt[total]+=1
        
        ans = sorted(cnt, reverse=True) # 降序
        return ans[:3]



grid = [[1,2,3],[4,5,6],[7,8,9]]
sol = Solution()
res = sol.getBiggestThree(grid)
print(res)