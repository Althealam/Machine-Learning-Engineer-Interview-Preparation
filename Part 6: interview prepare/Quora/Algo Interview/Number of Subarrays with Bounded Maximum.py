
class Solution:
    def numSubarrayBoundedMax(self, nums: list[int], left: int, right: int) -> int:
        return self.count(nums, right)-self.count(nums, left-1)

    def count(self, nums, bound):
        ans = 0
        cur = 0
        for x in nums:
            if x<=bound:
                cur+=1
            else:
                cur = 0
            ans+=cur
        return ans