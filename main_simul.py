
import numpy as np
import communicationsystem

#inp_file_dir = 'sample.txt'
inp_file_dir = 'Lenna.png'
source_coding_type = "Huffman"
draw_huffmantree = False
#source_coding_type = "NoCompression"
modulation_scheme = "BPSK"
mu = 0
std = 0.5

img_noise = communicationsystem.inp_with_noise(inp_file_dir,source_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)

pass

