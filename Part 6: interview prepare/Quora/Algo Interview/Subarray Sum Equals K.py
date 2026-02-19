# [1, 1, 1]: [1, 1], [1, 1]
# [1, 2, 3]: [1, 2], [3]

class Solution:
    def subarraySum(self, nums: List[int], k: int) -> int:
        prefix_sum = [0]*(len(nums)+1)
        for i, x in enumerate(nums):
            prefix_sum[i+1] = prefix_sum[i]+x
        
        ans = 0
        cnt = defaultdict(int)
        for prefix in prefix_sum:
            ans+=cnt[prefix-k]
            cnt[prefix]+=1
        return ans
        