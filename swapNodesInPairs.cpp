struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};
// 24 leet
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (head == nullptr || head->next == nullptr) {
            return head;
        }
        // head is 1
        ListNode* helper = head->next;
        ListNode* prev = nullptr;

        while(head and head->next) {
            if (prev != nullptr) {
                prev->next = head->next; // Step 4
            }
            prev = head;     
            // Get reference to 2's next (which is 3)
            ListNode* nextNode = head->next->next;

            // 2 points to 1
            head->next->next = head;
            // 1 points to 3
            head->next = nextNode;
            // iterate and jump 2
            head = nextNode;
        }
        return(helper);
    }
};