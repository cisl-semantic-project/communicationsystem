import os
import cv2
import numpy as np
def read_file_func(inp_file_dir):
    fname, ext = os.path.splitext(inp_file_dir)
    if ext == '.txt':
        with open(inp_file_dir, 'r+', encoding='UTF-8') as file:
            inp_data = file.read()
            frequency_dict = {}
            data_to_idx_dict = {}
            idx_to_data_dict = {}
            for character in inp_data:
                if not character in frequency_dict:
                    frequency_dict[character] = 0
                    idx = len(data_to_idx_dict)
                    data_to_idx_dict[character] =idx
                    idx_to_data_dict[idx] = character
                frequency_dict[character] += 1
            frequency_dict = dict(sorted(frequency_dict.items()))

            inp_data_np = np.array([[data_to_idx_dict[char] for char in inp_data]], dtype='uint32')
            mapped_data = np.copy(inp_data_np)

        bit_len = len(data_to_idx_dict).bit_length()
    else:
        inp_data = cv2.imread(inp_file_dir, cv2.IMREAD_COLOR)
        inp_data = cv2.cvtColor(inp_data, cv2.COLOR_BGR2GRAY)
        inp_data_np = np.copy(inp_data)

        bit_len = 8  # 이미지의 비트길이는 추후 수정해야할 듯
        frequency_dict = {}
        data_to_idx_dict = {}
        idx_to_data_dict = {}
        mapped_data_list = []
        for pix_array in inp_data_np:
            for pix_val in pix_array:
                if not pix_val in frequency_dict:
                    frequency_dict[pix_val] = 0
                    idx = len(data_to_idx_dict)
                    data_to_idx_dict[pix_val] = idx  # pix_value에 번호를 매김
                    idx_to_data_dict[idx] = pix_val
                mapped_data_list.append(data_to_idx_dict[pix_val])
                frequency_dict[pix_val] += 1
        assert len(data_to_idx_dict) < 65535, "최대 pix 갯수는 65535임."

        mapped_data = np.array(mapped_data_list,dtype = 'uint32').reshape(inp_data.shape)

    assert len(data_to_idx_dict) < 65535, "최대 word 갯수는 65535임."
    assert bit_len <= 32, "입력 데이터의 원소 별 비트 길이는 32비트보다 작아야함."

    return inp_data, inp_data_np, mapped_data, frequency_dict, data_to_idx_dict,idx_to_data_dict,bit_len, ext