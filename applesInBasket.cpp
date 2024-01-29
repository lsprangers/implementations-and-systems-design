#include <queue>;
#include <vector>;
using namespace std;
//1196

class Solution {
public:
    int maxNumberOfApples(vector<int>& weight) {
        int totalWeight = 5000;
        int resp = 0;
        priority_queue<int, vector<int>, greater<int>> minHeap(weight.begin(), weight.end());
        while(!minHeap.empty() && totalWeight >0) {
            int thisWeight = minHeap.top();
            minHeap.pop();
            if(thisWeight <= totalWeight) {
                totalWeight -= thisWeight;
                resp++;
            }
            else {
                break;
            }
        }
        return(resp);
    }
};