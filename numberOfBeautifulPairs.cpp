#include <vector>;
using namespace std;
//2748
class Solution {
public:
    int countBeautifulPairs(vector<int>& nums) {
        int resp = 0;
        for(int i = 0; i < nums.size(); i++) {
            for(int j = i + 1; j < nums.size(); j++) {
                int inum = nums[i];
                int jnum = nums[j] % 10;
                while(inum >= 10) {
                    inum /= 10;
                }
                if(gcd(inum, jnum) == 1) {
                    resp++;
                }
            }
        }
        return(resp);
    }
    int gcd(int a, int b) { 
        // Everything divides 0 
        if (a == 0) 
            return b; 
        if (b == 0) 
            return a; 
    
        // base case 
        if (a == b) 
            return a; 
    
        // a is greater 
        if (a > b) 
            return gcd(a - b, b); 
        return gcd(a, b - a); 
    }   
};