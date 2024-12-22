#include <set>;
#include <stack>;
#include <iostream>;
using namespace std;

class Solution {
public:
    bool isValid(string s) {
        stack<char> stack;
        set<char> ends{')', '}', ']'};
        unordered_map<char, char> lookup{ {'(', ')'},  {'[', ']'},  {'{', '}'} };

        for(char c : s) {
            // If end
            if(ends.find(c) != ends.end()) {
                if(stack.size() == 0) {
                    return(false);
                }
                char latest = stack.top();
                stack.pop();
                if(lookup[latest] != c) {
                    return(false);
                }
            }
            else {
                stack.push(c);
            }
        }
        return(stack.size() == 0);
    }
};