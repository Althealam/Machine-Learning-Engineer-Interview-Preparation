# [1, 2, 2, 2, 2] target = 5
# {1: 1, 2: 4}

# arr[x]+arr[y]+arr[z] = target ==> arr[y]+arr[z] = target-arr[x]
class Solution:
    def threeSumMulti(self, arr: List[int], target: int) -> int:
        MOD = 10**9+7
        count = Counter(arr)
        keys = sorted(count.keys())
        ans = 0

        for i, x in enumerate(keys):
            T = target - x
            for j in range(i, len(keys)):
                y = keys[j]
                z = T-y
                if x<y<z:
                    ans+=count[x]*count[y]*count[z]
                elif x==y<z:
                    ans+=count[x]*(count[x]-1)//2*count[z]
                elif x<y==z:
                    ans+=count[x]*(count[y]*(count[y]-1))//2
                elif x==y==z:
                    ans+=count[x]*(count[x]-1)*(count[x]-2)//6
        
        return ans%MOD
