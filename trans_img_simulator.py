import cv2
import communicationsystem
from math import log10
import between_std_SNR
def nothing(x):
    pass

inp_file_dir = 'Lenna.png'
source_coding_type = "Huffman"
source_coding_type = "NoCompression"
#channel_coding_type = "NoChannelCoding"
channel_coding_type = "Repetition"
draw_huffmantree = False      # huffman이 아니면 True여도 안그림.
modulation_scheme = "BPSK"
mu = 0
SNR = 5 #dB
std = between_std_SNR.SNR_2_std(SNR)


Tx_window = 'Tx input'
Rx_window = "Rx result"

track_bar_name = 'SNR트랙바'

cv2.namedWindow(Tx_window)
result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)
inp_image = result_class.inp_data
img_noise = result_class.out_data

cv2.imshow(Tx_window, inp_image)

cv2.namedWindow(Rx_window)
cv2.createTrackbar(track_bar_name, Rx_window, 2*SNR+20, 50, nothing)

blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255)
font =  cv2.FONT_HERSHEY_PLAIN

while(True):
    SNR = (cv2.getTrackbarPos(track_bar_name, Rx_window)-20)/2
    SNR_txt = "%.1fdB"%(SNR)
    std = between_std_SNR.SNR_2_std(SNR)

    result_class = communicationsystem.make_result_class(inp_file_dir, source_coding_type, channel_coding_type, draw_huffmantree,
                                                         modulation_scheme, mu, std)
    img_noise = result_class.out_data
    cv2.putText(img_noise, source_coding_type, (20, 40), font, 2, red, 3)
    cv2.putText(img_noise, channel_coding_type, (20, 80), font, 2, red, 3)
    cv2.putText(img_noise, SNR_txt, (20, 120), font, 2, red, 3)

    cv2.imshow(Rx_window, img_noise)

    if cv2.waitKey(100)&0xFF == 27:
        break

cv2.destroyAllWindows()