class Solution:
    def countGoodSubstrings(self, s: str) -> int:
        ans = 0
        list_s = list(s)
        for i in range(len(list_s)-1):
            sub_s = list_s[i:i+3]
            if len(Counter(sub_s))==3:
                ans+=1
        return ans