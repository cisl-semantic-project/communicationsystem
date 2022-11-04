import numpy as np
import cv2


def deci2bit(input,bit_len=None):
    def to_binary(number, max_len):
        '''
        int 데이터와 길이를 입력하면 입력 길이짜리 bit list를 반환한다.
        '''
        bits = "{0:b}".format(number, '#b')
        num_bits = len(bits)
        bit_list = list(map(int, bits))

        bit_list = (max_len - num_bits) * [0] + bit_list
        return bit_list

    '''
    넘파이의 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    output_list = []
    row, col = np.shape(input)
    if bit_len ==None:
        bit_len = int(np.max(input)).bit_length()

    for i in range(row):
        for j in range(col):
            bits_ = to_binary(int(input[i][j]), bit_len)
            output_list.append(bits_)
    return np.reshape(np.array(output_list), (-1, 1))
def mod_bpsk(inp_bit):
    '''
    입력 bit 넘파이가 0이면 -1 1이면 1로 변환하는 BPSK 모듈레이션
    '''
    out_sym = np.where(inp_bit == 0, -1, 1)
    return out_sym

def demod_bpsk(inp_sym):
    '''
    입력 넘파이가 0보다 크면 1, 작으면 -1로 변환하는 BPSK 디모듈레이션
    '''
    out_bit = (inp_sym > 0) + np.zeros(np.shape(inp_sym))
    return out_bit

def channel_awgn(inp_sym, mu=0, sigma=0.1):
    '''
    넘파이와 평균, 표준편차를 입력하면 awgn을 더해서 반환함
    '''
    output = inp_sym + np.random.normal(mu, sigma, np.shape(inp_sym))
    return output

'''
이미지 테스트 
'''
img_color = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
inp = img_gray

a = deci2bit(inp,8)
b = mod_bpsk(a)
c = demod_bpsk(b)
d = channel_awgn(b,0,1)
e = demod_bpsk(d)
np.reshpe(e,np.shape(inp))
pass


