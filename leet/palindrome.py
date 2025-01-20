# O(n) time for loops (O(n/2))
# O(1) space since we only use 2 integer variables and 2 pointers

def check_palindrome(s: str) -> bool:
    left = 0 
    right = len(s) - 1

    # O(n / 2) iterations == O(n) time
    while left < right:
        # O(1) random access
        if s[left] != s[right]:
            return False
        
        left += 1
        right -= 1
    
    return True

assert(check_palindrome('racecar') == True)
assert(check_palindrome('racebar') == False)
assert(check_palindrome('abcdcba') == True)
assert(check_palindrome('bbbbbbb') == True)