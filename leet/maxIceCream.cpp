// problem 1833
#include "vector"
using namespace std;

class Solution {
public:
    int maxIceCream(vector<int>& costs, int coins) {
        std::sort(costs.begin(), costs.end());
        int res = 0;
        for(int c : costs){
            if(coins >= c){
                coins -= c;
                res += 1;
            }
            else{
                break;
            }
        }
        return(res);
    }
};