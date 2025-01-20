#include <vector>;
#include <iostream>;
using namespace std;
//22 
class Solution {
public:
    int n;
    vector<string> generateParenthesis(int n) {
        vector<string> resp;
        string curr = "";
        this->n = n;
        backtrack(curr, resp, 0, 0);
        return(resp);
    }
    void backtrack(string& curr, vector<string>& resp, int left, int right) {
        if(curr.size() == 2*n) {
            resp.push_back(curr);
            return;
        }
        if(left < n) {
            backtrack(curr.append("("), resp, left + 1, right);
            curr.pop_back();
        }
        if(left > right) {
            backtrack(curr.append(")"), resp, left, right + 1);
            curr.pop_back();
        }
    }
};