#include <set>;
#include <stack>;
#include <iostream>;
using namespace std;

class Solution {
public:
    bool isValid(string s) {
        if(s.size() < 2) {
            return(false);
        }
        stack<char> stk;
        vector<char> right = {')', ']', '}'};
        vector<char> left = {'(', '[', '{'};
        
        for(int i = 0; i < s.size(); i++) {
            if( find(left.begin(), left.end(), s[i]) != left.end() ) {
                stk.push(s[i]);
            }
            else {
                if(stk.size() < 1) {
                    return(false);
                }
                char toCheck = stk.top();
                stk.pop();
                auto it = find(left.begin(), left.end(), toCheck); 
  
                if (it != left.end()) {
                    int idx = it - left.begin(); 
                    if(s[i] != right[idx]) {
                        return(false);
                    }
                }
            }
        }
        if(stk.size() > 0) {
            return(false);
        }
        return(true);
    }
};