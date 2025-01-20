//234
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    bool isPalindrome(ListNode* head) {
        vector<int> toCheck;
        while(head) {
            toCheck.push_back(head->val);
            head = head->next;
        }
        
        int n = toCheck.size() - 1;
        int left = 0;
        int right = n - left;
        while(left <= right) {
            if(toCheck[left] != toCheck[right]) {
                return(false);
            }
            left++;
            right = n - left;
        }
        return(true);

    }
};