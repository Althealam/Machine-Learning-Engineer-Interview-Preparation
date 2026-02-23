def solution1(nums, target):
    """all positive number in the nums and don't have duplicate number"""
    left = 0
    count = 0
    sum_ = 0
    for right in range(len(nums)):
        sum_+=nums[right]
        if sum_==target:
            count+=1
        while sum_>target:
            sum_-=nums[left]
            left+=1
    return count

def solution2(nums, k):
    """there are negative number in the nums"""
    prefix_sum = [0]*len(nums)
    for i in range(len(nums)):
        if i==0:
            prefix_sum[0] = nums[0]
        else:
            prefix_sum[i] = prefix_sum[i-1]+nums[i]
    
    hash = {}
    hash[0] = 1
    count = 0
    for j in range(len(nums)):
        current_prefix = prefix_sum[j]
        target_prefix = current_prefix-k
        if target_prefix in hash:
            count+=hash[target_prefix]
        hash[current_prefix] = hash.get(current_prefix, 0)+1
        # if current_prefix not in hash:
        #     hash[current_prefix]=1
        # else:
        #     hash[current_prefix]+=1
    return count