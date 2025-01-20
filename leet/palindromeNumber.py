# 9 
class Solution:
    def isPalindrome(self, x: int) -> bool:
        strx = str(x)
        lenx = len(strx)
        for i in range(lenx // 2):
            if strx[i] == strx[lenx - i - 1]:
                continue
            return False
        
        return True
        