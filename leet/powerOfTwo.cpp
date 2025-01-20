#include <math.h>;
using namespace std;
//231
class Solution {
public:
    bool isPowerOfTwo(int n) {
        return(fmod(log2(n), 1.0)==0.0);
    }
};