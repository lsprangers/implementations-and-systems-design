#include <map>
#include <string>
using namespace std;
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int start = 0;
        int maxLength = 0;
        map<char, int> usedChar;
        int str_len = s.length();

        for(int i = 0; i < str_len; i++){
            if(
                // if letter in usedChar and start less than index
                (usedChar.count(s[i]) > 0) && (start <= usedChar[s[i]])
            ){
                    // move start index to next string 
                    start = usedChar[s[i]] + 1;
            }
            else {
                maxLength = max(maxLength, i - start + 1);
            }
            usedChar[s[i]] = i;
        }
        return maxLength;

    }
};