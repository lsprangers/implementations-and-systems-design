class Solution:
    def allPathsSourceTarget(self, graph):
        def depthh(i, cur):
            if i == nodes - 1:
                paths.append(cur)
                return

            for j in graph[i]:
                depthh(j, cur + [j])

        nodes = len(graph)
        paths = []
        depthh(0, [0])
        return paths