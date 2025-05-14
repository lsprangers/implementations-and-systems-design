# 3
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        if len(s) < 1:
            return(0)

        curr = 1
        seen = set(s[0])
        left = 0
        right = 1
        while right < len(s):
            while seen and s[right] in seen:
                seen.remove(s[left])
                left += 1

            seen.update(s[right])
            right += 1
            curr = max(curr, right - left)
        
        return(curr)