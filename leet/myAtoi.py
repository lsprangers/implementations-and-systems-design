# 8 py
class Solution:
    def myAtoi(self, s: str) -> int:
        s = s.strip()
        if len(s) < 1:
            return 0
        neg = False
        if s[0] == "-":
            neg = True
            s = s[1:]
            if len(s) < 1:
                return 0
        if s[0] == "+":
            if neg == True:
                return 0
            neg = False
            s = s[1:]
        
        out=[]
        for ch in s:
            try:
                num_ch = int(ch)
                out.append(ch)
            except ValueError:
                break
        
        res = ''.join(out)
        if len(res) < 1:
            return 0
        res = int(res)
        if neg:
            res = -res
        if res < -2**31:
            res = -2**31
        if res > (2**31) - 1:
            res = (2**31) - 1
        return res