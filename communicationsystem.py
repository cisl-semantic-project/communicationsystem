import numpy as np
from read_file_func import read_file_func
from sourceencoder import huffman2

class communicationsystem:
    def __init__(self, ext,inp_data,inp_data_np,mapped_data,frequency_dict,data_to_idx_dict,idx_to_data_dict,
                 source_coding_type="NoCompression",inp_bit_len = None,
                 modulation_scheme = None,
                 mu = 0, std =1):

        ##source코딩에 필요한 파라미터
        self.ext = ext
        self.inp_data =inp_data
        self.inp_data_np =inp_data_np
        self.mapped_data =mapped_data
        self.frequency_dict = frequency_dict
        self.data_to_idx_dict = data_to_idx_dict
        self.idx_to_data_dict = idx_to_data_dict

        self.source_coding_type = source_coding_type
        self.inp_bit_len = inp_bit_len
        self.mu = mu
        self.std = std
        self.modulation_scheme = modulation_scheme

        self.data_to_code_dict = None
        self.code_to_data_dict = None
        self.source_coding_result_np = None
        self.source_coding_result_num_np = None
        self.modulation_result = None
        self.channel_result = None
        self.demodulation_result = None
        self.source_decoding_result_np = None
        self.source_decoding_result_approx_np = None
        self.out_data = None

def source_encoder(inp_class):
    '''
    넘파이 데이터와 각 원소의 비트수(입력안하면 가장 큰 비트로 맞춤)를 입력 받아서 [비트길이 x 1]형태의 비트 넘파이로 변환한다.
    '''
    columned_inp = inp_class.mapped_data.reshape(-1, 1)
    ######## source_encoder
    if inp_class.source_coding_type == "Huffman":
        h = huffman2.HuffmanCoding(columned_inp,inp_class.frequency_dict,inp_class.data_to_idx_dict)
        draw_graph = False
        source_coding_result_np,source_coding_result_num_np = h.compress(inp_class,draw_graph)

    elif inp_class.source_coding_type == "NoCompression":
        print("%d length source coding"%inp_class.inp_bit_len)

        source_coding_result_np = np.flip(np.unpackbits(columned_inp.view('uint8'), axis=1, count=inp_class.inp_bit_len,bitorder='little'))  # 데이터를 바이트로 나누고 비트로 변경함
        source_coding_result_num_np = np.array([inp_class.inp_bit_len]*columned_inp.size)

    else:
        raise Exception("압축 알고리즘 이름 확인 필요함.")

    inp_class.source_coding_result_np, inp_class.source_coding_result_num_np = source_coding_result_np, source_coding_result_num_np
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
        inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 2, np.nan, inp_class.source_coding_result_np)
        inp_class.modulation_result = np.where(inp_class.modulation_result == 0, -1, inp_class.modulation_result)
        #inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 1, 1, inp_class.source_coding_result_np)
        #inp_class.modulation_result = np.where(inp_class.source_coding_result_np == 1, 1, inp_class.modulation_result)
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
    '''
        구현해야함
    '''
    return bit_stream
def source_decoder(inp_class) :

    if inp_class.source_coding_type == "Huffman":
        inp_class.code_to_data_dict
        inp_class.demodulation_result

    elif inp_class.source_coding_type == "NoCompression":
        ########### 소스디코딩에서 활용하자
        demodulation_result = np.copy(inp_class.demodulation_result)
        if inp_class.mapped_data.dtype == "uint8":
            padding_num = 0
        elif inp_class.mapped_data.dtype == "uint16":
            padding_num = 16 - inp_class.inp_bit_len
        elif inp_class.mapped_data.dtype == "uint32":
            padding_num = 32 - inp_class.inp_bit_len
        else:
            assert False, "mapped_data 자료형 확인필요"

        demodulation_result = np.pad(np.flip(demodulation_result), ((0, 0), (0, padding_num)))
        source_decoding_result_np = np.packbits(demodulation_result, axis=1, bitorder='little').view(inp_class.mapped_data.dtype)

        #여기는 실제로 얻어진 넘파이
        inp_class.source_decoding_result_np = source_decoding_result_np.reshape(
            np.shape(inp_class.mapped_data))  # mapped data
        
        #여기는 인덱스를 근사한 넘파이
        max_idx = max(inp_class.idx_to_data_dict.keys())
        inp_class.source_decoding_result_approx_np = np.where(source_decoding_result_np> max_idx,max_idx,source_decoding_result_np).reshape(
            np.shape(inp_class.mapped_data))   #dictonary 최대 인덱스보다 큰 애들은 최대 인덱스로 매핑하는 근사.

        if inp_class.ext == ".txt":
            dec_res_inp_data_np = np.copy(inp_class.source_decoding_result_approx_np)  # inp_data_np

            u, inv = np.unique(dec_res_inp_data_np, return_inverse=True)
            dec_res_inp_data = "".join(list(np.array([inp_class.idx_to_data_dict[idx] for idx in u])[inv])) # inp_data
            inp_class.out_data = dec_res_inp_data
        elif inp_class.ext == ".png":

            u, inv = np.unique(inp_class.source_decoding_result_approx_np, return_inverse=True)
            dec_res_inp_data_np = np.array([inp_class.idx_to_data_dict[idx] for idx in u])[inv].reshape(inp_class.inp_data_np.shape)  # inp_data_np
            inp_class.out_data = np.copy(dec_res_inp_data_np)  # inp_data




def inp_with_noise(inp_file_dir,source_coding_type,modulation_scheme,mu,std):
    '''
    디지털통신 시스템에 입력값을 통과시키는 함수
    '''

    inp_data,inp_data_np, mapped_data, frequency_dict, data_to_idx_dict,idx_to_data_dict, bit_len, ext = read_file_func(inp_file_dir)

    inp_class = communicationsystem(ext, inp_data,inp_data_np,mapped_data,frequency_dict,data_to_idx_dict, idx_to_data_dict,
                                    source_coding_type,bit_len,
                                    modulation_scheme,
                                    mu,std)

    source_encoder(inp_class)

    modulation(inp_class)

    channel_awgn(inp_class)

    demodulation(inp_class)
    source_decoder(inp_class)

    np.array_equal(inp_class.mapped_data,  inp_class.source_decoding_result_approx_np)  # 데이터의 입 출력이 동일함을 확인할 수 있음.

    return inp_class.out_data


#inp_with_noise(inp,1,8)

