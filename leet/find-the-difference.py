class Solution:
    def findTheDifference(self, s: str, t: str) -> str:
        for sl in s:
            if sl in t:
                t = t.replace(sl, "", 1)
        return t