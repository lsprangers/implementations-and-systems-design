// 2593

class Solution {
public:
    long long findScore(vector<int>& nums) {
        priority_queue<pair<int, int>, vector<pair<int, int>>, greater<pair<int, int>>> minHeap;
        for(int i = 0; i < nums.size(); i++) {
            minHeap.push(pair(nums[i], i));
        }

        long long score = 0;
        set<int> seenIdxs;

        while(minHeap.size() > 0) {
            pair<int, int> smallestNumStruct = minHeap.top();
            minHeap.pop();
            if(seenIdxs.find(smallestNumStruct.second) == seenIdxs.end()) {
                score += smallestNumStruct.first;
                seenIdxs.insert(smallestNumStruct.second);
                seenIdxs.insert(smallestNumStruct.second - 1);
                seenIdxs.insert(smallestNumStruct.second + 1);
            }
        }
        return(score);
    }
};