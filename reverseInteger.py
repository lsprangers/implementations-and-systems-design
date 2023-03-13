# 7
class Solution:
    def reverse(self, x: int) -> int:
        neg=False
        str_x = str(x)
        if x < 0:
            neg=True  
            str_x = str_x[1:]
        str_y = str_x[::-1]
        y = int(str_y)
        if neg:
            y = -y
        if (y < (-2**31)) or (y > (2**31) - 1):
            y = 0
        return y