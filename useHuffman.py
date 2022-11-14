from huffman import HuffmanCoding

path = "sample2.txt"

h = HuffmanCoding(path)
draw_graph = True
output_path = h.compress(draw_graph)
print("Compressed file path: " + output_path)

decom_path = h.decompress(output_path)
print("Decompressed file path: " +  decom_path)