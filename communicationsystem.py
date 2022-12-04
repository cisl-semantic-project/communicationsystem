import numpy as np
from read_file_func import read_file_func
from sourceencoder import Huffman

class communicationsystem:
    def __init__(self, ext,inp_data,mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                 source_coding_type,channel_coding_type,inp_bit_len = None, draw_huffmantree = False,
                 modulation_scheme = None,
                 mu = 0, std =1):
        
        self.ext = ext                                                  # 입력 파일 타입. (txt,)
        self.inp_data =inp_data                                         # 입력 데이터. (781600...)
        self.inp_data_unique_arr = inp_data_unique_arr                  # 입력 데이터의 unique arr. (01678...)
        self.inp_data_unique_arr_idx_arr = inp_data_unique_arr_idx_arr  # 입력 데이터의 unique arr의 idx. (01234...), mapped_data와 관련있음.
        self.mapped_data =mapped_data                                   # 입력 데이터를 순서대로 매핑한 arr. (341200...)
        self.count = count                                              # 각 매핑데이터의 빈도 arr. (21111...)
        self.source_coding_type = source_coding_type                    # 소스코딩 타입. (Huffman,)
        self.channel_coding_type = channel_coding_type                  # 채널코딩 타입. (Repetition,)
        self.inp_bit_len = inp_bit_len                                  # 입력 비트 길이, txt면 inp_data_unique_arr로 결정, png면 8로 고정.
        self.draw_huffmantree = draw_huffmantree                        # tree 결과 그릴지 여부.
        self.mu = mu                                                    # 가우시안 분포의 평균.
        self.std = std                                                  # 가우시안 분포의 표준편차.
        self.modulation_scheme = modulation_scheme                      # 모듈레이션 타입. (BPSK,)

        self.mapped_data_bit_num = None                                 # mapped_data가 가진 bit 총 갯수, 소스코딩에따라 달라짐.
        self.code_arr = None                                            # mapped_data 각각이 가진 bit code,inp_data_unique_arr_idx_arr 와 순서 동일, 길이가 다를 경우 2가 포함되어있음.
        self.source_coding_result_np = None                             # 소스코딩 결과.
        self.source_coding_result_bit_num = None                        # 소스코딩 결과의 비트수, 2가 제외되어 계산되어있음.
        self.channel_coding_result_np = None                            # 채널코딩 결과.
        self.channel_coding_result_bit_num = None                       # 채널코딩 결과의 비트수, source_coding_result_bit_num로 계산함.
        self.modulation_result = None                                   # 모듈레이션 결과 2는 nan으로 바뀜.
        self.channel_result = None                                      # 채널겪고난 후 결과
        self.demodulation_result = None                                 # 디모듈레이션 결과. nan이 다시 2로 바뀜.
        self.channel_decoding_result_np = None                          # 채널 디코딩 결과.
        self.source_decoding_result_np = None                           # 소스 디코딩 결과.
        self.out_data = None                                            # 입력 데이터형태로 변경된 결과물.

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

def channel_coding(inp_class):
    if inp_class.channel_coding_type == "NoChannelCoding":
        inp_class.channel_coding_result_np = np.copy(inp_class.source_coding_result_np)
        inp_class.channel_coding_result_bit_num = inp_class.source_coding_result_bit_num
    elif inp_class.channel_coding_type == "Repetition":
        inp_class.channel_coding_result_np = np.tile(inp_class.source_coding_result_np, (3, 1, 1))
        inp_class.channel_coding_result_bit_num = inp_class.source_coding_result_bit_num * 3
    else:
        assert False,"channel_coding_type 확인해야함."
def modulation(inp_class):

    if inp_class.modulation_scheme == "BPSK":
        inp_class.modulation_result = np.where(inp_class.channel_coding_result_np  == 2, np.nan, inp_class.channel_coding_result_np)
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
def channel_decoding(inp_class):

    if inp_class.channel_coding_type == "NoChannelCoding":
        inp_class.channel_decoding_result_np = np.copy(inp_class.demodulation_result)
    elif inp_class.channel_coding_type == "Repetition":
        inp_class.channel_decoding_result_np = np.where(inp_class.demodulation_result==2,np.nan,inp_class.demodulation_result)

        inp_class.channel_decoding_result_np =np.sum(inp_class.channel_decoding_result_np, axis=0)

        inp_class.channel_decoding_result_np = np.where(inp_class.channel_decoding_result_np < 2, 0,
                                                        inp_class.channel_decoding_result_np)
        inp_class.channel_decoding_result_np = np.where(inp_class.channel_decoding_result_np>1,1,inp_class.channel_decoding_result_np)
        inp_class.channel_decoding_result_np = np.where(np.isnan(inp_class.channel_decoding_result_np),2,inp_class.channel_decoding_result_np).astype('uint8')

    else:
        assert False,"channel_coding_type 확인해야함."

def source_decoder(inp_class) :
    detection_result_np = np.copy(inp_class.channel_decoding_result_np)
    inp_class.source_decoding_result_np = np.copy(inp_class.mapped_data)
    if inp_class.source_coding_type == "Huffman":

        # 채널의 영향으로 깨진 데이터 idx 뭉탱이
        # 채널의 영향으로 깨진 데이터 뭉탱이 inp_class.channel_decoding_result_np[error_idx] 코드북내에 있어도 깨지면 깨진걸로 본다. 이건 생각해봐야할 문제.
        error_idx = np.where(np.all(inp_class.channel_decoding_result_np == inp_class.source_coding_result_np, axis=1)==False)[0]

        code_2num_in_codebook=(inp_class.code_arr == 2).sum(axis=1) #코드북내의 코드별 2의 갯수
        # u1: codebook의 2의 갯수 array, v1 : code 별 u1 idx
        #u1, v1 = np.unique(code_2num_in_codebook, return_inverse=True)

        # u3 : 2의 갯수 array, v3 : 깨진 데이터의 code 별 u3 idx, c3는 2의 갯수별 깨진 데이터의 갯수
        u3, v3, c3= np.unique((inp_class.channel_decoding_result_np[error_idx] == 2).sum(axis=1), return_inverse=True,return_counts=True)

        for i in range(u3.size):
            # 2의 갯수 별 codebook arr
            idx_arr = np.where(code_2num_in_codebook == u3[i])[0]

            random_idx_arr = np.random.randint(idx_arr.size, size=c3[i])
            # 깨진 데이터의 2의 갯수별 demoul결과의 array를 같은 2의 갯수의 코드로 랜덤하게 바꾸겠다.
            # refer_arr = inp_class.code_arr[idx_arr]
            #detection_result_np[error_idx][np.where(v3==i)] = refer_arr[np.random.randint(refer_arr.shape[0],size = c3[i])]

            # source decoding의 결과는 idx만 가져오는 결과
            inp_class.source_decoding_result_np[error_idx[np.where(v3==i)]] = idx_arr[random_idx_arr]

        '''  아래는 ML detection, 그러나 해당 방법은 너무 시간이 오래 걸린다. 
        u1, v1 = np.unique((inp_class.code_arr == 2).sum(axis=1),
                           return_inverse=True)  # u1: 2의 갯수 array, v1 : code 별 u1 idx
        u2,v2 = np.unique((inp_class.channel_decoding_result_np == 2).sum(axis=1),return_inverse=True)
        for i in u1: #2의 갯수가 가장작은것 부터 큰것까지 순회하겠음.
            code_idx_arr = np.where(v1 == i)[0]
            code_arr_with_2i = inp_class.code_arr[code_idx_arr].astype('int8') # 2의 갯수가 i개인 코드 어레이들 뭉탱이
            demodul_result_idx_arr = np.where(v2 == i)

            for demodul_result_idx in demodul_result_idx_arr[0]  : # 2의 갯수가 i개인 디모듈 어레이들 뭉탱이
                detection_result = np.argmin(np.power(inp_class.channel_decoding_result_np[demodul_result_idx].astype('int8') - code_arr_with_2i.astype('int8'), 2).sum(axis=1)) # bool로 하면 더 빨라질듯
                inp_class.source_decoding_result_np[demodul_result_idx] = code_idx_arr[detection_result] #mapped data 결과
        '''

        if inp_class.ext == ".txt":
            inp_class.out_data = "".join(list(inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np]))

        elif inp_class.ext == ".png":
            inp_class.out_data =inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np].reshape(inp_class.inp_data.shape)


    elif inp_class.source_coding_type == "NoCompression":
        channel_decoding_result_np = np.copy(inp_class.channel_decoding_result_np)
        source_decoding_result_np = np.packbits(channel_decoding_result_np, axis=1, bitorder='little').view(inp_class.mapped_data.dtype)
        last_idx = inp_class.inp_data_unique_arr_idx_arr[-1]
        inp_class.source_decoding_result_np = np.where(source_decoding_result_np>last_idx,last_idx,source_decoding_result_np).reshape(inp_class.mapped_data.shape) # idx가 넘는애들을 근사함.

        if inp_class.ext == ".txt":
            inp_class.out_data = "".join(list(inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np]))
        elif inp_class.ext == ".png":
            inp_class.out_data = inp_class.inp_data_unique_arr[inp_class.source_decoding_result_np].reshape(
                inp_class.inp_data.shape)

def make_result_class(inp_file_dir,source_coding_type,channel_coding_type,draw_huffmantree,modulation_scheme,mu,std):
    inp_data, mapped_data, inp_data_unique_arr,inp_data_unique_arr_idx_arr, count, inp_bit_len, ext = read_file_func(inp_file_dir)

    inp_class = communicationsystem(ext, inp_data, mapped_data,inp_data_unique_arr, inp_data_unique_arr_idx_arr,count,
                                    source_coding_type,channel_coding_type, inp_bit_len,draw_huffmantree,
                                    modulation_scheme,
                                    mu,std)

    source_encoder(inp_class)
    channel_coding(inp_class)
    modulation(inp_class)

    channel_awgn(inp_class)

    demodulation(inp_class)
    channel_decoding(inp_class)
    source_decoder(inp_class)

    return inp_class
