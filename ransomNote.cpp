#include <vector>;
#include <iostream>;
using namespace std;
//383
class Solution {
public:
    bool canConstruct(string ransomNote, string magazine) {
        unordered_map<char, int> mag;
        unordered_map<char, int> ransom;
        
        for(char r: ransomNote) {
            ransom[r] += 1;
            
        }
        for(char m: magazine) {
            mag[m] += 1;
            
        }
        for(auto [key, val] : ransom) {
            if(val > mag[key]) {
                return(false);
            }
        }
        return(true);
        
    }
};