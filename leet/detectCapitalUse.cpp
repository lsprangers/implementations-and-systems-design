// problem 520
#include "string"
using namespace std;

class Solution {
public:
    bool detectCapitalUse(string word) {
        int capCount = 0;
        for(char& c: word){
            if(isupper(c)){
                capCount++;
            }
        }

        // all upper or lower
        if(capCount == word.length() || capCount == 0){
            return(true);
        }
        // camel case
        else if(isupper(word[0]) && capCount == 1){
            return(true);
        }
        else {
            return(false);
        }
    }
};