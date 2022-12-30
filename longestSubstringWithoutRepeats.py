class Solution:
    # @return an integer
    def lengthOfLongestSubstring(self, s):
        # Basically just goes through string once, and will continue adding to length if next letter
        #   is not in the latest substring
        start = maxLength = 0
        usedChar = {}

        for i in range(len(s)):
            # If we've seen this character before and the start of our current substring 
            #   is at an earlier position we would re-configure where start is
            #       which basically means starting over using a new substring
            #       because the maxLength would be updated in else 
            # Also we need to check that start is less than entry of used char because if start
            #   is greater than usedChar entry of it that means the letter is apart of the new
            #   substring and the last time it was used it was in another substring
            if s[i] in usedChar and start <= usedChar[s[i]]:
                start = usedChar[s[i]] + 1
            
            else:
                # If we haven't seen this character before then we check if our
                #   old maxLength, or the substring we now have, is larger
                #       since i - start + 1 = length of currently un-repeated substring
                maxLength = max(maxLength, i - start + 1)

            # Lastly we update the usedChar dictionaries input of that char
            usedChar[s[i]] = i
            print(usedChar, start, maxLength)
        
        return maxLength
