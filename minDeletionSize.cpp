// problem 945
#include "vector"
using namespace std;

class Solution {
public:
    int minDeletionSize(vector<string>& strs) {
        int countt = 0;
        for(int j = 0; j < strs[0].length(); j++){
            for(int i = 0; i < strs.size() - 1; i++){
                if(strs[i + 1][j] < strs[i][j]){
                    countt++;
                    break;
                }  
            }
        }
        return countt;      
    }
};