import numpy as np
import math
import os
import cv2
from sourceencoder import huffman2
def source_encoder(inp_data : np.ndarray, frequency_dict : dict, char_to_idx_dict : dict
                   ,inp_bit_len : int = None) ->  np.ndarray:
    '''
    넘파이 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''

    ### source_encoder

    h = huffman2.HuffmanCoding(inp_data,frequency_dict,char_to_idx_dict)

    draw_graph = False
    result_np = h.compress(draw_graph)
    ###

    if inp_bit_len == None:
        inp_bit_len = int(np.max(inp_data)).bit_length()

    assert inp_bit_len >= int(np.max(inp_data)).bit_length(), '입력 비트길이가 데이터의 최대 값의 비트 길이보다 커야함'

    columned_inp = inp_data.reshape(-1, 1)  # 입력 넘파이를 컬럼 벡터로 변환해줌 size:[n x 1]
    inp_bit = np.unpackbits(columned_inp.view('uint8'),axis=1, count=inp_bit_len,bitorder='little') #데이터를 바이트로 나누고 비트로 변경함
    ########### 디모듈에서 활용가능
    if columned_inp.dtype =="uint8":
        a = np.packbits(inp_bit, axis=1, bitorder='little')
        a = a.reshape(np.shape(inp_data))
    elif columned_inp.dtype =="uint32":
        b = np.pad(inp_bit,((0,0),(0,32-inp_bit.shape[1])))
        a = np.packbits(b, axis=1, bitorder='little').view(columned_inp.dtype)
        a = a.reshape(np.shape(inp_data))
    else:
        assert False, "입력 데이터 타입 확인 필요함."
    result_data = columned_inp.reshape(np.shape(inp_data))
    np.array_equal(result_data, a) # 데이터의 입 출력이 동일함을 확인할 수 있음.
    ###########

    result_np = np.reshape(inp_bit, (-1, 1))
    return result_np
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
def make_noise(std, img):
    height, width = img.shape
    img_noise = np.zeros((height, width), dtype=np.float)
    for i in range(height):
        for a in range(width):
            make_noise = np.random.normal()  # 랜덤함수를 이용하여 노이즈 적용
            set_noise = std * make_noise
            img_noise[i][a] = img[i][a] + set_noise
    return img_noise
def make_dict_n_numpy(text):
    frequency_dict = {}
    char_to_idx_dict = {}
    idx = 0
    for character in text:
        if not character in frequency_dict:
            frequency_dict[character] = 0
        frequency_dict[character] += 1
    frequency_dict = dict(sorted(frequency_dict.items()))

    for char, count in frequency_dict.items():
        min_token_count = 0
        if count >= min_token_count: # 최소 카운트는 필요하면 적용가능
            char_to_idx_dict[char] = len(char_to_idx_dict) #vocab에 번호를 매기겠다.
    assert len(char_to_idx_dict) <65535,"최대 word 갯수는 65535임."
    out_np = np.array([[char_to_idx_dict[char] for char in text]],dtype='uint32')

    return out_np,frequency_dict,char_to_idx_dict

def inp_with_noise(inp_file_dir,std):
    '''
    디지털통신 시스템에 입력값을 통과시키는 함수
    '''
    fname, ext = os.path.splitext(inp_file_dir)

    if ext == '.txt':
        with open(inp_file_dir, 'r+', encoding='UTF-8') as file:
            text = file.read()
            text = text.rstrip()
            inp_numpy, frequency_dict, char_to_idx_dict = make_dict_n_numpy(text)
        bit_len = len(char_to_idx_dict).bit_length()
    else:
        inp_numpy = cv2.imread(inp_file_dir, cv2.IMREAD_COLOR)
        inp_numpy = cv2.cvtColor(inp_numpy, cv2.COLOR_BGR2GRAY)
        bit_len = 8
    assert bit_len <= 32, "입력 데이터의 원소 별 비트 길이는 32비트보다 작아야함."

    inp_numpy, frequency_dict, char_to_idx_dict
    result_sourcecoding = source_encoder(inp_numpy,frequency_dict,char_to_idx_dict,bit_len)

    #디코딩 시 필요할듯 char_to_idx_dict

    output_symbol = modulation(result_sourcecoding, scheme="BPSK")
    y = channel_awgn(output_symbol, 0, std)
    demod_output = demodulation(y)
    output_data = bit2data(demod_output, np.shape(inp_numpy))
    return output_data


#inp_with_noise(inp,1,8)

