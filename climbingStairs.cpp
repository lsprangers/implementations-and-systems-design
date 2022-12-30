// for any number n = [n-1] + [n-2]
//  because we can take either one or two steps at the beginning
//  & if we take 1 then it's the same solution as n-1, and if we 
//      take 2 it's the same solution as n-2
#include <vector>;

class Solution {
public:
    int climbStairs(int n) {
        std::vector< unsigned int > fibs;
        fibs.push_back(1);
        fibs.push_back(2);

        if (n < 3) {
            return fibs[n-1];
        }
        else {
            int i = 2;
            while(i < n+2) {
                fibs.push_back(fibs[i-2] + fibs[i-1]);
                i++;
            }
            return fibs[n-1];
        }
    }
};