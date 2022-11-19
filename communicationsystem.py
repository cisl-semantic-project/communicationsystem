import numpy as np
from read_file_func import read_file_func
from sourceencoder import huffman2

class communicationsystem:
    def __init__(self, inp_data,mapped_data,frequency_dict,data_to_idx_dict,
                 source_coding_type="NoCompression",inp_bit_len = None,
                 modulation_scheme = None,
                 mu = 0, std =1):

        ##source코딩에 필요한 파라미터
        self.inp_data =inp_data
        self.mapped_data =mapped_data
        self.frequency_dict = frequency_dict
        self.data_to_idx_dict = data_to_idx_dict
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
    columned_inp = inp_class.mapped_data.reshape(-1, 1)
    ######## source_encoder
    if inp_class.source_coding_type == "Huffman":
        h = huffman2.HuffmanCoding(columned_inp,inp_class.frequency_dict,inp_class.data_to_idx_dict)
        draw_graph = False
        source_coding_result_np,source_coding_result_num_np = h.compress(draw_graph)

    elif inp_class.source_coding_type == "NoCompression":
        print("%d length source coding"%inp_class.inp_bit_len)
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
        a = a.reshape(np.shape(inp_class.inp_data))
        np.array_equal(inp_class.inp_data, a)  # 데이터의 입 출력이 동일함을 확인할 수 있음.
        ########### 디모듈에서 활용하자
    else:
        raise Exception("압축 알고리즘 이름 확인 필요함.")
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
def inp_with_noise(inp_file_dir):
    '''
    디지털통신 시스템에 입력값을 통과시키는 함수
    '''

    inp_data, mapped_data, frequency_dict, data_to_idx_dict, bit_len = read_file_func(inp_file_dir)

    source_coding_type = "Huffman"
    #source_coding_type = "NoCompression"
    modulation_scheme = "BPSK"
    mu = 0
    std = 1
    inp_class = communicationsystem(inp_data,mapped_data,frequency_dict,data_to_idx_dict,
                                    source_coding_type,bit_len,
                                    modulation_scheme,
                                    mu,std)

    source_encoder(inp_class)

    modulation(inp_class)

    channel_awgn(inp_class)
    demodulation(inp_class)
    source_decoder(inp_class)
    return output_data


#inp_with_noise(inp,1,8)

