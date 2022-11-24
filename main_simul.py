
import numpy as np
import communicationsystem

#inp_file_dir = 'sample.txt'
inp_file_dir = 'Lenna.png'
#source_coding_type = "Huffman"
source_coding_type = "NoCompression"
draw_huffmantree = False #huffman이 아니면 True여도 안그림.


modulation_scheme = "BPSK"
mu = 0
std = 0.5

img_noise = communicationsystem.inp_with_noise(inp_file_dir,source_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)

pass

