import numpy as np
from read_file_func import read_file_func
from sourceencoder import Huffman
class communicationsystem:
    def __init__(self, ext,inp_data,mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                 source_coding_type="NoCompression",inp_bit_len = None, draw_huffmantree = False,
                 modulation_scheme = None,
                 mu = 0, std =1):

        ##source코딩에 필요한 파라미터
        self.ext = ext
        self.inp_data =inp_data
        self.mapped_data =mapped_data
        self.mapped_data_bit_num = None
        self.count = count
        self.inp_data_unique_arr = inp_data_unique_arr
        self.inp_data_unique_arr_idx_arr = inp_data_unique_arr_idx_arr
        self.code_arr = None
        self.source_coding_type = source_coding_type
        self.inp_bit_len = inp_bit_len
        self.draw_huffmantree = draw_huffmantree
        self.mu = mu
        self.std = std
        self.modulation_scheme = modulation_scheme

        self.source_coding_result_np = None
        self.source_coding_result_bit_num = None
        self.modulation_result = None
        self.channel_result = None
        self.demodulation_result = None
        self.source_decoding_result_np = None
        self.out_data = None

def source_encoder(inp_class):

    if inp_class.source_coding_type == "Huffman":
        h = Huffman.HuffmanCoding(inp_class.mapped_data,inp_class.count,inp_class.inp_data_unique_arr,inp_class.inp_data_unique_arr_idx_arr , inp_class.draw_huffmantree)
        source_coding_result_np,code_arr,mapped_data_to_code_bit_num = h.compress()

        inp_class.mapped_data_bit_num =inp_class.mapped_data.size * inp_class.inp_data_unique_arr_idx_arr.size.bit_length()
        inp_class.source_coding_result_bit_num = np.sum(mapped_data_to_code_bit_num)

    elif inp_class.source_coding_type == "NoCompression":
        inp_class.mapped_data.reshape(-1,)
        source_coding_result_np = np.unpackbits(inp_class.mapped_data.reshape(-1,1).view('uint8'), axis=1, count=inp_class.inp_bit_len,bitorder='little')  # 데이터를 바이트로 나누고 비트로 변경함
        code_arr = np.unpackbits(inp_class.inp_data_unique_arr_idx_arr.reshape(-1, 1), axis=1, count=inp_class.inp_bit_len,
                      bitorder='little')

        inp_class.mapped_data_bit_num = inp_class.mapped_data.size * inp_class.inp_bit_len
        inp_class.source_coding_result_bit_num = np.sum(source_coding_result_np.size)

    else:
        raise Exception("압축 알고리즘 이름 확인 필요함.")

    inp_class.source_coding_result_np = source_coding_result_np
    inp_class.code_arr = code_arr

def channel_coding(bit_stream):
    return bit_stream
def modulation(inp_class):

    if inp_class.modulation_scheme == "BPSK":
        inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 2, np.nan, inp_class.source_coding_result_np)
        inp_class.modulation_result = np.where(inp_class.modulation_result == 0, -1, inp_class.modulation_result)
    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_awgn(inp_class):
    inp_class.channel_result = inp_class.modulation_result + np.random.normal(inp_class.mu, inp_class.std, inp_class.modulation_result.shape)
def demodulation(inp_class):
    if inp_class.modulation_scheme == "BPSK":
        inp_class.demodulation_result = np.where(inp_class.channel_result < 0, 0,
                                               inp_class.channel_result)
        inp_class.demodulation_result = np.where(inp_class.demodulation_result  > 0, 1,
                                               inp_class.demodulation_result)
        inp_class.demodulation_result = np.where(np.isnan(inp_class.demodulation_result),2,inp_class.demodulation_result).astype('uint8')

    else:
        raise Exception('모듈레이션 scheme 확인필요')
def channel_decoding(bit_stream):

    return bit_stream
def source_decoder(inp_class) :

    if inp_class.source_coding_type == "Huffman":

        inp_class.source_decoding_result_np = np.zeros_like(inp_class.mapped_data)

        u1,v1 = np.unique((inp_class.code_arr == 2).sum(axis=1),return_inverse=True) # 2의 갯수 array, code 별 2의 갯수,
        u2,v2 = np.unique((inp_class.demodulation_result == 2).sum(axis=1),return_inverse=True)

        for i in u1: #2의 갯수가 가장작은것 부터 큰것까지 순회하겠음.
            code_idx_arr = np.where(v1 == i)[0]
            code_arr_with_2i = inp_class.code_arr[code_idx_arr].astype('int8') # 2의 갯수가 i개인 코드 어레이들 뭉탱이
            demodul_result_idx_arr = np.where(v2 == i)

            for demodul_result_idx in demodul_result_idx_arr[0]  : # 2의 갯수가 i개인 디모듈 어레이들 뭉탱이
                detection_result = np.argmin(np.power(inp_class.demodulation_result[demodul_result_idx].astype('int8') - code_arr_with_2i.astype('int8'), 2).sum(axis=1)) # bool로 하면 더 빨라질듯
                inp_class.source_decoding_result_np[demodul_result_idx] = code_idx_arr[detection_result] #mapped data 결과


        if inp_class.ext == ".txt":
            inp_class.out_data = "".join(list(inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np]))

        elif inp_class.ext == ".png":
            inp_class.out_data =inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np].reshape(inp_class.inp_data.shape)


    elif inp_class.source_coding_type == "NoCompression":
        demodulation_result = np.copy(inp_class.demodulation_result)
        source_decoding_result_np = np.packbits(demodulation_result, axis=1, bitorder='little').view(inp_class.mapped_data.dtype)
        last_idx = inp_class.inp_data_unique_arr_idx_arr[-1]
        inp_class.source_decoding_result_np = np.where(source_decoding_result_np>last_idx,last_idx,source_decoding_result_np).reshape(inp_class.mapped_data.shape) # idx가 넘는애들을 근사함.

        if inp_class.ext == ".txt":
            inp_class.out_data = "".join(list(inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np]))
        elif inp_class.ext == ".png":
            inp_class.out_data = inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np].reshape(
                inp_class.inp_data.shape)

def make_result_class(inp_file_dir,source_coding_type,draw_huffmantree,modulation_scheme,mu,std):
    inp_data, mapped_data, inp_data_unique_arr,inp_data_unique_arr_idx_arr, count, bit_len, ext = read_file_func(inp_file_dir)

    inp_class = communicationsystem(ext, inp_data, mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                                    source_coding_type,bit_len,draw_huffmantree,
                                    modulation_scheme,
                                    mu,std)

    source_encoder(inp_class)
    modulation(inp_class)
    channel_awgn(inp_class)
    demodulation(inp_class)
    source_decoder(inp_class)

    return inp_class
