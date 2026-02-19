class Solution:
    def encode(self, num: int) -> str:
        binary_num_plus = self.get_binary(num+1)
        return ''.join(str(x) for x in binary_num_plus[1:])
        
    def get_binary(self, n):
        res = []
        while n!=0:
            res.append(n%2)
            # print("current n:", n)
            n//=2
        return res[::-1]