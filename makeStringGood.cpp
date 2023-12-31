#include <iostream>;
using namespace std;
// 1544 leet

class Solution {
public:
    string makeGood(string s) {
        if(s.size() < 2) {
            return(s);
        }
        string ans = "";
        
        for(int i = 0; i < s.size(); i++) {
            if(!ans.empty()) {
                char lastChar = ans.back();
                if( (toupper(lastChar) == s[i] or toupper(s[i]) == lastChar) and lastChar != s[i] ) {
                    ans.pop_back();
                    continue;
                }                
            }
            ans.push_back(s[i]);
        }
        return(ans);
    }
};