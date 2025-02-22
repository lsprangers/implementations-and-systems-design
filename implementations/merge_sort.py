# 0, 1, 2, 3, 4
# mid = 0 + (4 - 0) // 2 == 2
# low = 0, high = 2,
#   0, 1, 2
#   mid = 0 + (2 - 0) // 2 == 1
#   low = 0, high = 1
#       mid = 0 + (1 - 0) // 2 = 0
#       low = 0, high = 0 --> return [0]
#   low = 2, high = 2 --> return[2]
# low = 3, high = 4
#   3, 4
#   mid = 3 + (4 - 3) // 2 == 2
#   low = 3, high = 2
#   low = 4, high = 4

class mergeSort:
    def __init__(self, arr):
        self.arr = arr
    
    def sort(self, low, high):
        if not self.arr:
            return(self.arr)
        
        if low >= high:
            return([self.arr[low]])
        
        mid = low + (high - low) // 2
        
        # can optimize this to be in place later
        arr1 = self.sort(low, mid)
        arr2 = self.sort(mid + 1, high)
        
        arr3 = self.merge(arr1, arr2)
        return(arr3)
        
        
    
    
    def merge(self, arr1, arr2):
        if(len(arr1) == 0):
            return(arr2)
        if(len(arr2) == 0):
            return(arr1)
        
        oneIdx = 0
        twoIdx = 0
        
        # can optimize this to be in place later
        resp = []
        while oneIdx < len(arr1) or twoIdx < len(arr2):
            if oneIdx < len(arr1) and twoIdx < len(arr2):
                if arr1[oneIdx] <= arr2[twoIdx]:
                    resp.append(arr1[oneIdx])
                    oneIdx += 1
                else:
                    resp.append(arr2[twoIdx])
                    twoIdx += 1
                    
            elif oneIdx < len(arr1):
                resp.append(arr1[oneIdx])
                oneIdx += 1
            else: 
                resp.append(arr2[twoIdx])
                twoIdx += 1

        return(resp)


if __name__ == "__main__":
    arr = [4,2,7,1,9,0]
    test = mergeSort(arr)
    assert(test.sort(0, len(arr) - 1) == [0,1,2,4,7,9])
    print(test.sort(0, len(arr) - 1))
    
    arr = [4,2,7,-1,9,0]
    test = mergeSort(arr)
    assert(test.sort(0, len(arr) - 1) == [-1,0,2,4,7,9])
    print(test.sort(0, len(arr) - 1))
    
    arr = [0]
    test = mergeSort(arr)
    assert(test.sort(0, len(arr) - 1) == [0]) 
    print(test.sort(0, len(arr) - 1))
    
    arr = []
    test = mergeSort(arr)
    assert(test.sort(0, len(arr) - 1) == [])   
    print(test.sort(0, len(arr) - 1))         