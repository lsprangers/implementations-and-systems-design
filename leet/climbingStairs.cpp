// 0, 1, 2, 3, 5 -- fibonacci seq
#include <vector>;
using namespace std;
// 70
class Solution {
public:
    int climbStairs(int n) {
        vector<int> tabb;
        tabb.push_back(0);
        tabb.push_back(1);
        tabb.push_back(2);
        for(int i = 3; i <=n; i++) {
            tabb.push_back(
                tabb[i - 1] // 1 step
                +
                tabb[i - 2] // 2 steps
            );
        }
        return(tabb[n]);
    }
};