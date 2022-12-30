class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {       
        int nums1_size = nums1.size();
        int nums2_size = nums2.size();
        vector<int> nums3;
        while(!nums1.empty() || !nums2.empty()){
            if(nums1.empty()){
                nums3.push_back(nums2.front());
                nums2.erase(nums2.begin());
            }
            else if(nums2.empty()){
                nums3.push_back(nums1.front());
                nums1.erase(nums1.begin());
            }
            else if(nums1.front() <= nums2.front()){
                nums3.push_back(nums1.front());
                nums1.erase(nums1.begin());
            }
            else{
                nums3.push_back(nums2.front());
                nums2.erase(nums2.begin());
            }
        }
        int arr_size = nums3.size();
        double medn = 0.0;
        for (int i: nums3)
            cout << i << ' ';
        if((arr_size % 2) == 0){
            medn = (medn + nums3[(arr_size / 2) - 1] + nums3[(arr_size / 2)]) / 2;
        }
        else {
            medn = medn + nums3[(arr_size / 2)];
        }
        return medn;
    }
};