#include <string>;
using namespace std;
//2864
class Solution {
public:
    string maximumOddBinaryNumber(string s) {
        int nOnes = 0;
        int sSize = s.size();
        string resp;
        for(char c : s) {
            if(c == '1') {
                nOnes++;
            }
        }
        while(nOnes > 1) {
            sSize--;
            nOnes--;
            resp.append("1");
        }
        while(sSize > 1) {
            sSize--;
            resp.append("0");
        }
        resp.append("1");
        return(resp);
    }
};