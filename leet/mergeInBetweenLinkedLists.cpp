//1669

struct ListNode {
    int val;
    ListNode *next;
    ListNode() : val(0), next(nullptr) {}
    ListNode(int x) : val(x), next(nullptr) {}
    ListNode(int x, ListNode *next) : val(x), next(next) {}
};

class Solution {
public:
    ListNode* mergeInBetween(ListNode* list1, int a, int b, ListNode* list2) {
        ListNode* dummy = new ListNode(0, list1);
        ListNode* helper1 = new ListNode(0, list1);
        helper1 = helper1->next;
        b = b - a;

        while(a > 1) {
            list1 = list1->next;
            helper1 = helper1->next;
            a--;
        }
        helper1 = helper1->next;

        while(b > 0) {
            helper1 = helper1->next;
            b--;
        }
        helper1 = helper1->next;
        
        list1->next = list2;
        while(list1->next != nullptr) {
            list1 = list1->next;
        }
        list1->next = helper1;
        return(dummy->next);

    }
};