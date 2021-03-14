import struct

fileName = "data/si2_b03_m400.B50"
saveFile = "data/si2_b03_m400_1.B50.out"
with open(fileName, "rb") as binaryText:
    # 参考资料上给的转化方法
    # nodes_In_Binary = binaryText.read(2)
    # nodes = struct.unpack("H", nodes_In_Binary)[0]
    # for i in range(nodes):
    #     for j in range(nodes):
    #         matrix = np.array([[0 for _ in range(nodes)] for _ in range(nodes)])
    # for node in range(nodes):
    #     edges_In_Binary = binaryText.read(2)
    #     edges = struct.unpack("H", edges_In_Binary)[0]
    #     for edge in range(edges):
    #         target_In_Binary = binaryText.read(2)
    #         target = struct.unpack("H", target_In_Binary)[0]
    #         matrix[node][target] = 1

    # 实际操作时用的数据
    nodes_In_Binary = binaryText.read(2)
    nodes = struct.unpack("H", nodes_In_Binary)[0]
    matrix = []
    for i in range(1, nodes+1):
        matrix.append([str(i)])

    for node in range(nodes):
        edges_In_Binary = binaryText.read(2)
        edges = struct.unpack("H", edges_In_Binary)[0]
        for edge in range(edges):
            target_In_Binary = binaryText.read(2)
            target = struct.unpack("H", target_In_Binary)[0]
            matrix[node].append(str(target+1))

maxLength = max([len(i) for i in matrix])
print(maxLength)
result = ""
for elements in matrix:
    while len(elements) < maxLength:
        elements.append(str(0))
    result += ",".join(elements) + ", "

result = result[:-2]

with open(saveFile, 'w') as file_object:
    file_object.write(result)


