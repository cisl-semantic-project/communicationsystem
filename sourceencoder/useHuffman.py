"""
출처
author: Bhrigu Srivastava
website: https:bhrigu.me
https://github.com/bhrigu123/huffman-coding
https://zephyrus1111.tistory.com/132
"""

from huffman import HuffmanCoding

path = "sample.txt"

h = HuffmanCoding(path)
draw_graph = True
output_path = h.compress(draw_graph)
print("Compressed file path: " + output_path)

decom_path = h.decompress(output_path)
print("Decompressed file path: " +  decom_path)