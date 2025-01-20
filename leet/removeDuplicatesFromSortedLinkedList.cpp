
struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(head==nullptr)return head;

        ListNode* checker=head;
        
        while(checker and checker->next) {
            if (checker->next->val == checker->val) {
                checker->next = checker->next->next;
            }
            else {
                checker = checker->next;
            }
            // slow = slow->next;
        }
        return(head);
    }
};