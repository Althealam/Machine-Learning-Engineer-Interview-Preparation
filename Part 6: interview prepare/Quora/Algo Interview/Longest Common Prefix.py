class Solution:
    def longestCommonPrefix(self, strs: List[str]) -> str:
        ans = strs[0]
        for i, ch in enumerate(ans):
            for st in strs:
                if i==len(st) or st[i]!=ans[i]:
                    return ans[:i]
        return ans

strs =["ab", "a"]
sol = Solution()
res = sol.longestCommonPrefix(strs)
print(res)