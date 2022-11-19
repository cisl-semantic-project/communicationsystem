import numpy as np
import math
import os
import cv2
from sourceencoder import huffman2

class communicationsystem:
    def __init__(self, inp_data,frequency_dict,char_to_idx_dict,
                 source_coding_type="NoCompression",inp_bit_len = None,
                 modulation_scheme = None,
                 mu = 0, std =1):
        ##source코딩에 필요한 파라미터
        self.inp_data =inp_data
        self.frequency_dict = frequency_dict
        self.char_to_idx_dict = char_to_idx_dict
        self.source_coding_type = source_coding_type
        self.inp_bit_len = inp_bit_len
        self.mu = mu
        self.std = std
        self.modulation_scheme = modulation_scheme

        self.source_coding_result_np = None
        self.source_coding_result_num_np = None
        self.modulation_result = None
        self.channel_result = None

        self.output_data = None

def source_encoder(inp_class):
    '''
    넘파이 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    columned_inp = inp_class.inp_data.reshape(-1, 1)
    ######## source_encoder
    if inp_class.source_coding_type == "Huffman":
        h = huffman2.HuffmanCoding(columned_inp,inp_class.frequency_dict,inp_class.char_to_idx_dict)
        draw_graph = False
        source_coding_result_np,source_coding_result_num_np = h.compress(draw_graph)
    elif inp_class.source_coding_type == "NoCompression":
        source_coding_result_np = np.flip(np.unpackbits(columned_inp.view('uint8'), axis=1, count=inp_class.inp_bit_len,bitorder='little'))  # 데이터를 바이트로 나누고 비트로 변경함
        source_coding_result_np = np.reshape(source_coding_result_np, (-1, 1))
        source_coding_result_num_np = np.array([inp_class.inp_bit_len]*columned_inp.size)
        ########### 디모듈에서 활용하자
        b=source_coding_result_np.reshape(-1, inp_class.inp_bit_len)
        if columned_inp.dtype == "uint8":
            padding_num = 0
        elif columned_inp.dtype == "uint16":
            padding_num = 16 - inp_class.inp_bit_len
        elif columned_inp.dtype == "uint32":
            padding_num = 32 - inp_class.inp_bit_len
        else:
            raise Exception("입력 데이터 자료형 확인필요")
        b = np.pad(np.flip(b), ((0, 0), (0, padding_num)))
        a = np.packbits(b, axis=1, bitorder='little').view(columned_inp.dtype)
        a = a.reshape(np.shape(inp_class.inp_numpy))
        np.array_equal(inp_class.inp_numpy, a)  # 데이터의 입 출력이 동일함을 확인할 수 있음.
        ########### 디모듈에서 활용하자
    else:
        raise Exception("압축 알고리즘 이름 확인 필요함.")

    assert inp_class.inp_bit_len >= np.max(source_coding_result_num_np), '입력 비트길이가 데이터의 최대 값의 비트 길이보다 커야함'
    ########
    inp_class.source_coding_result_np = source_coding_result_np
    inp_class.source_coding_result_num_np = source_coding_result_num_np
def channel_coding(bit_stream):
    '''
        구현해야함
    '''
    return bit_stream
def modulation(inp_class):
    '''
    데이터와 비트 길이, scheme을 입력에 따라 symbol을 반환
    '''
    def mod_bpsk(inp_bit):
        '''
        입력 bit 넘파이가 0이면 -1 1이면 1로 변환하는 BPSK 모듈레이션
        '''
        out_sym = np.where(inp_bit == 0, -1, 1)

    if inp_class.modulation_scheme == "BPSK":
        inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 0, -1, 1)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_awgn(inp_class):
    '''
    넘파이와 평균, 표준편차를 입력하면 awgn을 더해서 반환함
    '''
    inp_class.channel_result = inp_class.modulation_result + np.random.normal(inp_class.mu, inp_class.std, inp_class.modulation_result.size)

def demodulation(inp_class):
    if inp_class.modulation_scheme == "BPSK":
        inp_class.demodulation_result = (inp_class.modulation_result>0).astype("uint8")
    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_decoding(bit_stream):
    '''
        구현해야함
    '''
    return bit_stream
def source_decoder(inp_class) :

    inp_class.output_data = None

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
            inp_data, frequency_dict, char_to_idx_dict = make_dict_n_numpy(text)
        bit_len = len(char_to_idx_dict).bit_length()
    else:
        inp_data = cv2.imread(inp_file_dir, cv2.IMREAD_COLOR)
        inp_data = cv2.cvtColor(inp_data, cv2.COLOR_BGR2GRAY)
        bit_len = 8
    assert bit_len <= 32, "입력 데이터의 원소 별 비트 길이는 32비트보다 작아야함."

    source_coding_type = "Huffman"
    #source_coding_type = "NoCompression"
    modulation_scheme = "BPSK"
    mu = 0
    std = 1
    inp_class = communicationsystem(inp_data,frequency_dict,char_to_idx_dict,
                                    source_coding_type,bit_len,
                                    modulation_scheme,
                                    mu,std)

    source_encoder(inp_class)
    modulation(inp_class)

    channel_awgn(inp_class)
    demodulation(inp_class)
    source_decoder(demod_output, np.shape(inp_numpy))
    return output_data


#inp_with_noise(inp,1,8)

