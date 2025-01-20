# 12

class Solution:
    def romanToInt(self, s: str) -> int:
        converter = {
            'I': 1,
            'V' : 5,
            'X' : 10,
            'L' : 50,
            'C' : 100,
            'D' : 500,
            'M' : 1000
        }
        num = 0
        if 'IV' in s:
            num += 4
            s = s.replace('IV', '')
        if 'IX' in s:
            num += 9
            s = s.replace('IX', '')
        if 'XL' in s:
            num += 40
            s = s.replace('XL', '')
        if 'XC' in s:
            num += 90
            s = s.replace('XC', '')
        if 'CD' in s:
            num += 400
            s = s.replace('CD', '')
        if 'CM' in s:
            num += 900
            s = s.replace('CM', '')
        
        for roman, val in converter.items():
            num += s.count(roman) * val
            s.replace(roman, '')
        return num