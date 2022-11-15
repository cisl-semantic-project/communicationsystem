import numpy as np
import cv2
import math
import sourcecoding
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
def bit2data(inp_bit, size=None):
    cnt1 = 0
    cnt2 = 0
    tmp = 0
    result = []
    result_tmp = []
    a = int(np.size(inp_bit) / (size[0]*size[1]))  # a:bits 수

    for i in range(np.shape(inp_bit)[0]):
        if inp_bit[i][0] == 1:
            tmp += math.pow(2, a-cnt1-1)
        cnt1 += 1
        if cnt1 == a:
            result_tmp.append(int(tmp))
            cnt2 += 1
            cnt1 = 0
            tmp = 0

        if cnt2 == size[1]:
            result.append(result_tmp)
            result_tmp = []
            cnt2 = 0
    return np.array(result)

def cal_ber(origin, recon):
    error = sum(origin.flatten() != recon.flatten())
    ber = error/(np.size(origin))
    return ber

'''
이미지 테스트 
'''
def inp_with_noise(inp,std,bit_len):
    output_bit = data2bit(inp, bit_len)
    output_symbol = modulation(output_bit, scheme="BPSK")
    y = channel_awgn(output_symbol, 0, std)
    demod_output = demodulation(y)
    output_data = bit2data(demod_output, np.shape(inp))
    return output_data

#inp_with_noise(inp,1,8)

