//3032

class Solution {
public:
    bool check(int tmp_a) {
        set<int> num;

        while(tmp_a > 0) {
            int next_dig = tmp_a % 10;
            if(num.find(next_dig) != num.end()) {
                return(false);
            }                
            num.insert(next_dig);
            tmp_a /= 10;
        }
        return(true);
    }
    int numberCount(int a, int b) {
        int resp = 0;
        
        for(int i = a; i <= b; i++) {
            resp += check(i) ? 1 : 0;
        }
        return(resp);
    }
};