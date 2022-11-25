
import numpy as np
import communicationsystem
import cv2

#inp_file_dir = 'sample.txt'
inp_file_dir = 'Lenna.png'
#source_coding_type = "Huffman" # detection 방법이 잘못되서 전체적으로 어두운 결과가 나옴
source_coding_type = "NoCompression"
draw_huffmantree = False      # huffman이 아니면 True여도 안그림.
modulation_scheme = "BPSK"
mu = 0
std = 1

result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)

cv2.imwrite('Test_dir/Test1.png', result_class.inp_data)
cv2.imwrite('Test_dir/Test2.png', result_class.out_data)
pass

