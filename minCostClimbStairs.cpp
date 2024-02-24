#include <vector>;
using namespace std;
// 746
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        int n = cost.size();
        vector<int> memo = vector(n + 1, 0);
        for(int i = 2; i <= n; i++) {
            memo[i] = min(memo[i - 1] + cost[i - 1], memo[i - 2] + cost[i - 2]);
        }
        return(memo[n]);

    }
};