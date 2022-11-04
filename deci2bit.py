import numpy as np


def to_binary(number):
    answer = "{0:b}".format(number, '#b')
    return answer, len(answer)

def deci2bit(input):
    bits = []
    row, col = np.shape(input)
    _, max_len = to_binary(np.max(input))

    for i in range(row):
        for j in range(col):
            bits_, len_ = to_binary(input[i][j])
            if len_ < max_len:
                bits_ = bits_.zfill(max_len)

            bits.append(bits_)
    return bits


inp = np.array([[1, 255, 0], [1, 100, 200]])
print(inp)
print(deci2bit(inp))
