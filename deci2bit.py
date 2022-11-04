import numpy as np
import cv2

def data2bit(inp,bit_len=None):
    '''
        넘파이의 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    def to_binary(number, max_len):
        '''
        int 데이터와 길이를 입력하면 입력 길이짜리 bit list를 반환한다.
        '''
        bits = "{0:b}".format(number, '#b')
        num_bits = len(bits)
        bit_list = list(map(int, bits))

        bit_list = (max_len - num_bits) * [0] + bit_list
        return bit_list

    output_list = []
    row, col = np.shape(inp)
    if bit_len ==None:
        bit_len = int(np.max(inp)).bit_length()

    for i in range(row):
        for j in range(col):
            bits_ = to_binary(int(inp[i][j]), bit_len)
            output_list.append(bits_)
    return np.reshape(np.array(output_list), (-1, 1))
def source_coding(bit_stream) :
    '''
    구현해야함
    '''
    return bit_stream
def channel_coding(bit_stream):
    '''
        구현해야함
    '''
    return bit_stream
def modulation(inp, scheme = "BPSK"):
    '''
    데이터와 비트 길이, scheme을 입력에 따라 symbol을 반환
    '''
    def mod_bpsk(inp_bit):
        '''
        입력 bit 넘파이가 0이면 -1 1이면 1로 변환하는 BPSK 모듈레이션
        '''
        out_sym = np.where(inp_bit == 0, -1, 1)
        return out_sym
    if scheme == "BPSK":
        mod_output = mod_bpsk(inp)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
    return mod_output
def channel_awgn(inp_sym, mu=0, sigma=0.1):
    '''
    넘파이와 평균, 표준편차를 입력하면 awgn을 더해서 반환함
    '''
    output = inp_sym + np.random.normal(mu, sigma, np.shape(inp_sym))
    return output
def demodulation(inp, scheme = "BPSK"):
    def demod_bpsk(inp_sym):
        '''
        입력 넘파이가 0보다 크면 1, 작으면 -1로 변환하는 BPSK 디모듈레이션
        '''
        out_bit = (inp_sym > 0) + np.zeros(np.shape(inp_sym))
        return out_bit

    if scheme == "BPSK":
        demod_output = demod_bpsk(inp)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
    return demod_output
def channel_decoding(bit_stream):
    '''
        구현해야함
    '''
    return bit_stream
def source_decoding(bit_stream) :
    '''
        구현해야함
    '''
    return bit_stream
def bit2data(inp):
    '''
        구현해야함
    '''
    return

'''
이미지 테스트 
'''
img_color = cv2.imread('Lenna.png', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
inp = img_gray

output_bit = data2bit(inp, 8)
output_symbol = modulation(output_bit,scheme = "BPSK")
y = channel_awgn(output_symbol,0,1)
demod_output = demodulation(y)

output_data = bit2data(output_bit,8)

pass


