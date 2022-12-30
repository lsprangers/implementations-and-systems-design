def longestPalindrome(s: str) -> str:
    n = len(s)
    maxL = 1
    mems = [[0 if i != j else 1 for i in range(n) ] for j in range(n)]

    # Check for len of 2 
    start = 0
    i = 0
    while i < n - 1:
        if(s[i] == s[i + 1]):
            mems[i][i+1] = 1
            start = i
            maxL = 2
        i += 1

    # Check for 3   
    # k is the length of substring we are checking - so we loop from 3 to n to check those length substrings
    # then we shift i and j k units at each step and 
    #   then when we check i+1 j-1 it's checking to see if the middle substring is a palindrome, and 
    #   then if s[i]==s[j] to check the letters outside of that...basically if the middle substring is palindrome, 
    #       and the outside letters are the same then it's another palindrome

    # Think of i as the lagging variable and j as the leading
    #   so that if i+1 and j-1 is palindrome it means the substring inbetween them
    #   and then we check s[i==j] which would be the outside letters
    k = 3
    while k <= n :
        i = 0
        # i goes up to n
        while i < (n - k + 1) :
            # j goes up to n-1
            j = i + k - 1   
            if (mems[i + 1][j - 1] and s[i] == s[j]): 
                mems[i][j] = 1
                if (k > maxL):
                    start = i
                    maxL = k
            i += 1
        k += 1

    return(s[start:start + maxL])

if __name__ == "__main__":
    out = longestPalindrome(s="bababdd")
    print(out)