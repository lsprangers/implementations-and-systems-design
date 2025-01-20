#include <string>;
using namespace std;
//1347 

class Solution {
public:
    int minSteps(string s, string t) {
        unordered_map<char, int> ss;
        unordered_map<char, int> ts;
        int resp = 0;
        for(char cs : s) {
            ss[cs]++;
        }
        for(char ct : t) {
            if(ss.find(ct) != ss.end()) {
                ss[ct]--;
                if(ss.at(ct) == 0) {
                    ss.erase(ct);
                }
            }
            else{
                resp++;
            }
        }
        return(resp);
    }
};