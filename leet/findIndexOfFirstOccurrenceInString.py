# 28
class Solution:
    def strStr(self, haystack: str, needle: str) -> int:
        for hIdx in range(len(haystack)):
            left = hIdx
            right = hIdx + len(needle)
            if right <= len(haystack) and haystack[left : right] == needle:
                return(hIdx)
        
        return(-1)