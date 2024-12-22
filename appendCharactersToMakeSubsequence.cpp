// 2486
class Solution {
public:
    int appendCharacters(string s, string t) {
        int s_left = 0;
        int t_left = 0;
        while(t_left < t.size() && s_left < s.size()) {
            if(t[t_left] == s[s_left]) {
                t_left++;
            }
            s_left++;
        }
        return(t.size() - t_left);
    }
};