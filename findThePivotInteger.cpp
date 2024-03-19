// 2485
class Solution {
public:
    int pivotInteger(int n) {
        int sumAll = (n * (n + 1)) / 2;

        for(int i = 0; i <= n; i++) {
            int sumX = (i * (i + 1)) / 2;
            int pivotSum = sumAll - sumX + i;
            
            if(pivotSum == sumX) {
                return(i);
            }
        }
        return(-1);
    }
};