// 58
class Solution {
public:
    int lengthOfLastWord(string s) {
        bool foundWord = false;
        int resp = 0;
        for(int i = s.size() - 1; i >= 0; i--) {
            if(s[i] == ' ' && foundWord) {
                break;
            }
            if(s[i] != ' ') {
                foundWord = true;
                resp++;
            }
        }
        return(resp);
    }
};