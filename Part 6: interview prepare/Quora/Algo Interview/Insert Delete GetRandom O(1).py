import random
class RandomizedSet:
    def __init__(self):
        self.nums = []
        self.indices = {}
        

    def insert(self, val: int) -> bool:
        if val in self.indices:
            return False
        self.indices[val] = len(self.nums)
        self.nums.append(val)
        return True

        
    def remove(self, val: int) -> bool:
        if val not in self.indices:
            return False
        
        idx_to_delete = self.indices[val]
        idx_last_element = self.indices[self.nums[-1]]

        # exchange place in nums
        self.nums[idx_to_delete], self.nums[-1] = self.nums[-1], val

        # exchange index self.indices
        self.indices[val], self.indices[self.nums[idx_to_delete]] = idx_last_element, idx_to_delete

        # delete
        self.nums.pop()
        del self.indices[val]

        return True
        

    def getRandom(self) -> int:
        if len(self.nums)!=0:
            return random.choice(self.nums)
        


# Your RandomizedSet object will be instantiated and called as such:
# obj = RandomizedSet()
# param_1 = obj.insert(val)
# param_2 = obj.remove(val)
# param_3 = obj.getRandom()