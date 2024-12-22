// 1860 
class Solution {
public:
    vector<int> memLeak(int memory1, int memory2) {
        int time_size = 1;

        while(max(memory1, memory2) >= time_size) {
            // cout << time_size << " " << memory1 << " " << memory2 << endl;
            if(memory1 >= memory2) {
                memory1 -= time_size;
            }
            else {
                memory2 -= time_size;
            }
            time_size++;
        }
        vector<int> resp = {time_size, memory1, memory2};
        return(resp);

    }
};