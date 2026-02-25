
class ExamRoom:
    def __init__(self, n: int):
        self.n = n
        self.seats = []
        
    def seat(self) -> int:
        # print('===================')
        # print("doing the seat function")
        if len(self.seats)==0:
            res = 0 # let the people sit on the place 0
        elif len(self.seats)==1: # there are only one people sit in the exam room
            left_dist = self.seats[0] 
            # left index is 0
            # right index is n-1
            right_dist = self.n-1-self.seats[0]
            if left_dist>right_dist:
                res = 0
            else:
                res = self.n-1   
        else:
            max_dist = self.seats[0] # 0到第一个人的最远距离
            res = 0
            for i in range(1, len(self.seats)): # iterate all the seat
                pre = self.seats[i-1]
                cur = self.seats[i]
                # print(f"Current cur is {cur}, current pre is {pre}")
                dist = (cur-pre)//2 # the new place distance between these two people
                if dist>max_dist:
                    max_dist = dist
                    res = pre+dist # update the new place 
            # check the edge condition
            if (self.n-1)-self.seats[-1]>max_dist:
                res = self.n-1

        # print(f"current place is {res}")
        # we have to ensure that self.seats is sorted
        self.seats.append(res) # put the res into self.seats
        self.seats.sort() # make sure it is sorted
        return res


    def leave(self, p: int) -> None:
        if p in self.seats:
            self.seats.remove(p)