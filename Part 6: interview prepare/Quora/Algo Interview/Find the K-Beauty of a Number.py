class Solution:
    def divisorSubstrings(self, num: int, k: int) -> int:
        cnt = 0
        for i in range(len(str(num))-k+1):
            sub_num = str(num)[i:i+k]
            # print(f"current sub num is {sub_num}")
            if int(sub_num)!=0 and num%int(sub_num)==0:
                cnt+=1
        return cnt

num = 2
k = 1
sol = Solution()
res = sol.divisorSubstrings(num, k)
print(res)

# class Solution:
#     def divisorSubstrings(self, num: int, k: int) -> int:
#         M = 10**k
#         ans = 0
#         n = num
#         while n>=M//10:
#             x = n%M
#             if x>0 and num%x==0:
#                 ans+=1
#             n//=10
#         return ans

# # time: O(n-k)
# # space: O(a)