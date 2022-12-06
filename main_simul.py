
import numpy as np
import communicationsystem
import between_std_SNR
import cv2

#inp_file_dir = 'sample.txt'
inp_file_dir = 'Lenna.png'
#source_coding_type = "Huffman"
source_coding_type = "NoCompression"
#channel_coding_type = "Repetition"
channel_coding_type = "NoChannelCoding"

draw_huffmantree = False      # huffman이 아니면 True여도 안그림.
modulation_scheme = "BPSK"
mu = 0
SNR = 0 #dB
std = between_std_SNR.SNR_2_std(SNR)

result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)
#####압축률 확인
result_class.source_coding_result_bit_num / result_class.mapped_data_bit_num
'''
    수학적 압축률(엔트로피 / mapped_data_bit_len) == 허프만 코딩의 바운드임(더 좋을 수 없다.)
    prob_arr = result_class.count/np.sum(result_class.count)
    -np.sum(prob_arr * np.log2(prob_arr))/ prob_arr.size.bit_length()
'''
#####


#####BER 확인
source_coding_result_bit_num = result_class.source_coding_result_bit_num #보내는 비트수
channel_coding_result_bit_num = result_class.channel_coding_result_bit_num

#원래 BER
err_bit_num = np.sum(result_class.channel_coding_result_np != result_class.demodulation_result)#에러난 비트수
BER = err_bit_num/channel_coding_result_bit_num
print("%.2fdB, %.4f "%(SNR, BER))

#채널 코딩으로 복원되서 더 좋아진 BER
err_bit_num =np.sum(result_class.source_coding_result_np !=result_class.channel_decoding_result_np) #에러난 비트수
recovered_BER = err_bit_num/source_coding_result_bit_num
print("%.2fdB, %.4f "%(SNR, recovered_BER))
'''
BER, recovered_BER, math_BER, math_BER_with_channel_coding_3_rept
수학적 BPSK의  BER => 평균적인 결과라서 바운드가 아님 (더 좋을 수 있다.)
from scipy import special
math_BER = 0.5 - 0.5*special.erf(1/(std*np.sqrt(2)))
math_BER_with_channel_coding_3_rept = 3*math_BER**2-2*math_BER**3 
'''
#####

#result_class.inp_data, result_class.out_data
#cv2.imwrite('Test_dir/Test1.png', result_class.inp_data)
#cv2.imwrite('Test_dir/Test2.png', result_class.out_data)
pass

