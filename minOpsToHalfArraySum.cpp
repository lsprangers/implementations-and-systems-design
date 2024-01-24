#include <vector>;
#include <queue>;
using namespace std;

//2208
class Solution {
public:
    int halveArray(vector<int>& nums) {
        priority_queue<double> heap(nums.begin(), nums.end());
        double initial_sum = 0.0;
        for(double num : nums) {
            initial_sum += num;
        }
        double curr_sum = initial_sum;
        int n_ops = 0;

        while(curr_sum > (initial_sum / 2)) {
            double top = heap.top();
            heap.pop();
            double halfer = top / 2;
            curr_sum -= halfer;
            heap.push(halfer);
            n_ops++;
        }
        return(n_ops);

    }
};