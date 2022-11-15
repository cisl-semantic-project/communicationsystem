import numpy as np
import math
import sourceencoder
def source_encoder(inp_data : np.ndarray, inp_bit_len : int = None) ->  np.ndarray:
    '''
    구현 목표
    1. 입력 데이터가 txt파일일 때, numpy.ndarry에다가 int형으로 매핑해야함.
    '''

    '''
    넘파이의 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    if inp_bit_len == None:
        inp_bit_len = int(np.max(inp_data)).bit_length()

    assert inp_bit_len >= int(np.max(inp_data)).bit_length(), '입력 비트길이가 데이터의 최대 값의 비트 길이보다 커야함'

    columned_inp = inp_data.reshape(-1, 1)  # 입력 넘파이를 컬럼 벡터로 변환해줌 size:[n x 1]
    inp_bit = np.unpackbits(columned_inp, axis=1, count=inp_bit_len)

    ########### 디모듈에서 활용가능
    b = np.packbits(inp_bit, axis=1)
    columned_inp = columned_inp.reshape(np.shape(inp_data))
    np.array_equal(columned_inp, inp_data)
    ###########
    ### source_encoder

    ###
    return np.reshape(inp_bit, (-1, 1))
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
    '''
    디지털통신 시스템에 입력값을 통과시키는 함수
    '''
    result_sourcecoding = source_encoder(inp,bit_len)
    output_symbol = modulation(result_sourcecoding, scheme="BPSK")
    y = channel_awgn(output_symbol, 0, std)
    demod_output = demodulation(y)
    output_data = bit2data(demod_output, np.shape(inp))
    return output_data

#inp_with_noise(inp,1,8)

