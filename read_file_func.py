import os
import cv2
import numpy as np
def read_file_func(inp_file_dir):
    fname, ext = os.path.splitext(inp_file_dir)
    if ext == '.txt':
        with open(inp_file_dir, 'r+', encoding='UTF-8') as file:
            inp_data = file.read()
            inp_data_np_arr = np.array(list(inp_data))
            inp_data_unique_arr, mapped_data, count = np.unique(inp_data_np_arr, return_inverse=True,
                                                                            return_counts=True)
        inp_bit_len = inp_data_unique_arr.size.bit_length()

        if inp_bit_len <=8:
            to_dtype = "uint8"
        elif inp_bit_len <=16:
            to_dtype = "uint16"
        elif inp_bit_len <=32:
            to_dtype = "uint32"
        elif inp_bit_len <= 64:
            to_dtype = "uint64"
        else:
            assert False, "데이터 한 원소의 비트길이가 너무 김."

        mapped_data = np.array(list(mapped_data), dtype=to_dtype)
        inp_data_unique_arr_idx_arr = np.unique(mapped_data)

    else:
        inp_data = cv2.imread(inp_file_dir, cv2.IMREAD_COLOR)
        inp_data = cv2.cvtColor(inp_data, cv2.COLOR_BGR2GRAY)

        inp_bit_len = 8  # 이미지의 비트길이는 추후 수정해야할 듯

        inp_data_unique_arr, mapped_data, count = np.unique(inp_data, return_inverse=True, return_counts=True)
        mapped_data = np.array(list(mapped_data),dtype='uint8')

        inp_data_unique_arr_idx_arr = np.unique(mapped_data)

    return inp_data, mapped_data, inp_data_unique_arr, inp_data_unique_arr_idx_arr, count, inp_bit_len, ext