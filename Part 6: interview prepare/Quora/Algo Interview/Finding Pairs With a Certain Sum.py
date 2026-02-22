class FindSumPairs:

    def __init__(self, nums1: List[int], nums2: List[int]):
        self.nums1 = nums1
        self.nums2 = nums2
        self.hash2 = Counter(self.nums2)

    def add(self, index: int, val: int) -> None:
        # update hash map 
        self.hash2[self.nums2[index]]-=1
        self.nums2[index]+=val
        self.hash2[self.nums2[index]]+=1
        
    def count(self, tot: int) -> int:
        # can not put the hash into count function, otherwise every time we call the count function will need lots of time 
        cnt = 0
        for i in range(len(self.nums1)):
            if tot-self.nums1[i] in self.hash2:
                cnt+=self.hash2[tot-self.nums1[i]]
        return cnt





# Your FindSumPairs object will be instantiated and called as such:
# obj = FindSumPairs(nums1, nums2)
# obj.add(index,val)
# param_2 = obj.count(tot)