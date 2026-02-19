# {12: [0, 9]}
# objective: find the frequence of value in the [left, right]
# find the first direction which is larger than the left index in the subarray, the first direction which is smaller than the right index in the subarray
# Example: [0, 9] is the subarray, [2, 3] is the left and right
# return 0 cause left_new and right_new is None
# return length of the subarray right_new-left_new+1

class RangeFreqQuery:
    def __init__(self, arr: List[int]):
        self.pos = defaultdict(list)

        for index, num in enumerate(arr):
            self.pos[num].append(index)

    def query(self, left: int, right: int, value: int) -> int:        
        if value not in self.pos:
            return 0
        arr = self.pos[value]
        l = self.bisect_left(arr, left)
        r = self.bisect_right(arr, right)
        return r-l

    def bisect_left(self, arr, target):
        # find the first direction which is larger than target
        # arr: [1, 3, 5] target = 4 return: 5
        l, r = 0, len(arr)
        while l<r:
            mid = (l+r)//2
            if arr[mid]<target:
                l=mid+1
            else:
                r=mid
        return l


    def bisect_right(self, arr, target):
        # find the first direction which is larger than right
        # arr: [1, 3, 5] target = 4 return: 5
        l, r = 0, len(arr)
        while l<r:
            mid = (l+r)//2
            if arr[mid]<=target:
                l = mid+1
            else:
                r = mid
        return l