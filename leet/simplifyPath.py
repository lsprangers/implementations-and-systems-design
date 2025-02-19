#71

class Solution:
    def simplifyPath(self, path: str) -> str:
        if path == "/" or path == "":
            return("/")
        
        components = path.split("/")
        resp_list = []
        for item in components:
            if item == "" or item == ".":
                continue

            elif item == "..":
                if len(resp_list) > 0:
                    resp_list.pop()
            else:
                resp_list.append(item)
        
        return(
            "/" + "/".join(resp_list)
        )