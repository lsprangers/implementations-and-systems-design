#include <queue>;
using namespace std;
//703
class KthLargest {
public:
    priority_queue<int, vector<int>, greater<int>> minHeap;
    // priority_queue<int> minHeap;
    int K;
    KthLargest(int k, vector<int>& nums) {
        K = k;
        for(int num : nums) {
            minHeap.push(num);
        }     
    }
    
    int add(int val) {
        minHeap.push(val);
        while(minHeap.size() > K) {
            minHeap.pop();
        }
        return(minHeap.top());
    }
};
