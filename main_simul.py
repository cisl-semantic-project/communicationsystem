
import numpy as np
import communicationsystem

inp_file_dir = 'Lenna.png'
#source_coding_type = "Huffman"
source_coding_type = "NoCompression"
modulation_scheme = "BPSK"
mu = 0
std = 0

img_noise = communicationsystem.inp_with_noise(inp_file_dir,source_coding_type,modulation_scheme,
                                               mu, std)

pass

