import cv2
import communicationsystem
from math import log10
def nothing(x):
    pass



inp_file_dir = 'Lenna.png'
#source_coding_type = "Huffman" #detection하는 방법에서 Complexity가 너무 높음 수정해야함.
source_coding_type = "NoCompression"
draw_huffmantree = False      # huffman이 아니면 True여도 안그림.
modulation_scheme = "BPSK"
mu = 0
std = 1

Tx_window = 'Tx input'
Rx_window = "Rx result"

track_bar_name = '표준편차= value/50'

cv2.namedWindow(Tx_window)
result_class = communicationsystem.make_result_class(inp_file_dir,source_coding_type,draw_huffmantree,
                                               modulation_scheme, mu, std)
inp_image = result_class.inp_data
img_noise = result_class.out_data

cv2.imshow(Tx_window, inp_image)

cv2.namedWindow(Rx_window)
cv2.createTrackbar(track_bar_name, Rx_window, std, 200, nothing)

blue = (255, 0, 0)
green= (0, 255, 0)
red= (0, 0, 255)
white= (255, 255, 255)
font =  cv2.FONT_HERSHEY_PLAIN

while(True):
    std = cv2.getTrackbarPos(track_bar_name, Rx_window) / 50
    if std==0 :
        SNR_txt = "infinity"
    else:
        SNR_txt = "%.1fdB"%(10 * log10(1 / (2 * std ^ 2)))

    result_class = communicationsystem.make_result_class(inp_file_dir, source_coding_type, draw_huffmantree,
                                                         modulation_scheme, mu, std)
    img_noise = result_class.out_data
    cv2.putText(img_noise, SNR_txt, (380, 40), font, 2, red, 3)

    cv2.imshow(Rx_window, img_noise)

    if cv2.waitKey(100)&0xFF == 27:
        break

cv2.destroyAllWindows()