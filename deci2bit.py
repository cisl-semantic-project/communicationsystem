import numpy as np

#, max_len
def to_binary(number, max_len):
    bit_list = []
    num_bits = number.bit_length()
    bits = "{0:b}".format(number, '#b')
    for i in range(num_bits):
        if bits[i] == '0':
            bit_list.append(0)
        else:
            bit_list.append(1)

    if num_bits < max_len:
        for i in range(max_len-num_bits):
            bit_list.insert(0, 0)

    return bit_list

def deci2bit(input):
    input_bits = []
    row, col = np.shape(input)
    max_len = int(np.max(input)).bit_length()

    for i in range(row):
        for j in range(col):
            bits_ = to_binary(int(input[i][j]), max_len)
            input_bits.append(bits_)
    return np.reshape(np.array(input_bits), (-1, 1))


inp =  np.random.randint(100, 130, size=(9, 10))

print(inp)
print(deci2bit(inp))
f = open("C:/Users/Jaehoon/Desktop/test.txt", 'w')
f.write(str(deci2bit(inp)))
f.close()



