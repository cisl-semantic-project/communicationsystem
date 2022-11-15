"""
출처
author: Bhrigu Srivastava
website: https:bhrigu.me
https://github.com/bhrigu123/huffman-coding
https://zephyrus1111.tistory.com/132
"""
import heapq
import os
import matplotlib.pyplot as plt
import seaborn as sns
class Tree:
	def __init__(self, root):
		assert root.isRoot, 'node should be specified as root'
		self.__root = root

	def getRoot(self):
		return self.__root

	def getLengthOfBranch(self, node, cnt=1):
		if node.parentNode is None:
			return cnt
		else:
			cnt += 1
			return self.getLengthOfBranch(node.parentNode, cnt)

	def getDepth(self, remove_Leaf=False):
		all_nodes = self.traverseInOrder()
		if remove_Leaf:
			depth = max([self.getLengthOfBranch(node) for node in all_nodes if not node.isLeaf])
		else:
			depth = max([self.getLengthOfBranch(node) for node in all_nodes])
		return depth

	def traverseInOrder(self, node=None):
		if node is None:
			node = self.__root
		res = []
		if node.leftChild != None:
			res = res + self.traverseInOrder(node.leftChild)
		res.append(node)
		if node.rightChild != None:
			res = res + self.traverseInOrder(node.rightChild)
		return res

	def drawTree(self):
		def drawNode(node, ax):
			if node is not None:
				if node.isLeaf:
					bbox = dict(boxstyle='round', fc='white')
					out_txt = node.code
				else:
					bbox = dict(boxstyle='square', fc=colors[node.getLevel() - 1], pad=1)
				## 텍스트 표시
				if node.char !=None:
					out_txt = node.char +" : "+str(node.freq)+", "+out_txt
				else :
					out_txt = str(node.freq)
				ax.text(node.x, node.y, out_txt, bbox=bbox, fontsize=10, ha='center', va='center')
				if node.parentNode is not None:  ## 부모 노드와 자식 노드 연결
					ax.plot((node.parentNode.x, node.x), (node.parentNode.y, node.y), color='k')
				drawNode(node.leftChild, ax)
				drawNode(node.rightChild, ax)
		root = self.__root
		x_coords = []
		y_coords = []
		for i, nd in enumerate(self.traverseInOrder()):
			nd.x = i
			nd.y = -(nd.getLevel() - 1)
			x_coords.append(nd.x)
			y_coords.append(nd.y)

		min_x, max_x = min(x_coords), max(x_coords)
		min_y, max_y = min(y_coords), max(y_coords)

		colors = sns.color_palette('hls', self.getDepth() - 1)
		fig = plt.figure(figsize=((max_x-min_x),1.5*(max_y-min_y)))
		renderer = fig.canvas.get_renderer()
		ax = fig.add_subplot()

		ax.set_xlim(min_x - 1, max_x + 1)
		ax.set_ylim(min_y - 0.5, max_y + 0.5)
		drawNode(root, ax)
		plt.rcParams['axes.unicode_minus'] = False
		plt.savefig('result_huffman.png')

class HuffmanCoding:
	def __init__(self, path):
		self.path = path
		self.heap = []
		self.codes = {}
		self.reverse_mapping = {}
		self.tree = None

	class HeapNode:
		def __init__(self, char, freq):
			self.char = char
			self.freq = freq
			self.x = None
			self.y = None
			self.isRoot = False
			self.parentNode = None
			self.leftChild = None
			self.rightChild = None
			self.isLeaf = (char!=None)
			self.code = ""
		def getLevel(self, cnt=1):
			if self.isRoot:
				return cnt
			else:
				cnt += 1
				cnt = self.parentNode.getLevel(cnt)
				return cnt

		def setLeftChild(self, node):
			self.leftChild = node
			node.parentNode = self

		def setRightChild(self, node):
			self.rightChild = node
			node.parentNode = self

		# defining comparators less_than and equals
		def __lt__(self, other):
			return self.freq < other.freq

		def __eq__(self, other):
			if(other == None):
				return False
			if(not isinstance(other, HeapNode)):
				return False
			return self.freq == other.freq

	# functions for compression:

	def make_frequency_dict(self, text):
		frequency = {}
		for character in text:
			if not character in frequency:
				frequency[character] = 0
			frequency[character] += 1
		return frequency

	def make_heap(self, frequency):
		for key in frequency:
			node = self.HeapNode(key, frequency[key]) #
			heapq.heappush(self.heap, node) # 최소힙으로 만들어짐. 기준은 freq

	def merge_nodes(self,drawing_onoff=False):
		while(len(self.heap)>1):
			node1 = heapq.heappop(self.heap)
			node2 = heapq.heappop(self.heap)
			merged = self.HeapNode(None, node1.freq + node2.freq)
			###
			merged.left = node1
			merged.right = node2
			merged.setLeftChild(node1)
			merged.setRightChild(node2)
			heapq.heappush(self.heap, merged)
		merged.isRoot = True
		self.tree = Tree(merged)

	def make_codes_helper(self, root, current_code): #재귀함수로 char로 이루어진 node 뭉탱이 class를 코드로 만들어줌. 되게 잘짰음.
		if(root == None):
			return

		if(root.char != None):
			self.codes[root.char] = current_code
			self.reverse_mapping[current_code] = root.char
			root.code = current_code
			return

		self.make_codes_helper(root.leftChild, current_code + "0")
		self.make_codes_helper(root.rightChild, current_code + "1")


	def make_codes(self):
		root = heapq.heappop(self.heap) #HeapNode 클래스
		current_code = ""
		self.make_codes_helper(root, current_code)

	def get_encoded_text(self, text):
		encoded_text = ""
		for character in text:
			encoded_text += self.codes[character]
		return encoded_text


	def pad_encoded_text(self, encoded_text):
		extra_padding = 8 - len(encoded_text) % 8
		for i in range(extra_padding):
			encoded_text += "0"

		padded_info = "{0:08b}".format(extra_padding)
		encoded_text = padded_info + encoded_text
		return encoded_text

	def get_byte_array(self, padded_encoded_text):
		if(len(padded_encoded_text) % 8 != 0):
			print("Encoded text not padded properly")
			exit(0)

		b = bytearray()
		for i in range(0, len(padded_encoded_text), 8):
			byte = padded_encoded_text[i:i+8]
			b.append(int(byte, 2))
		return b


	def compress(self,draw_graph = False):
		filename, file_extension = os.path.splitext(self.path)
		output_path = filename + ".bin"

		with open(self.path, 'r+',encoding='UTF-8') as file, open(output_path, 'wb') as output:
			text = file.read()
			text = text.rstrip()

			frequency = self.make_frequency_dict(text)
			self.make_heap(frequency) #
			self.merge_nodes(True) #여기서 huffman coding에서 볼 수 있는  tree를 생성함. True를 통해 허프만 결과 저장가능
			self.make_codes()
			if draw_graph:
				self.tree.drawTree()

			encoded_text = self.get_encoded_text(text)
			padded_encoded_text = self.pad_encoded_text(encoded_text)

			b = self.get_byte_array(padded_encoded_text)
			output.write(bytes(b))

		print("Compressed")
		return output_path


	""" functions for decompression: """
	def remove_padding(self, padded_encoded_text):
		padded_info = padded_encoded_text[:8]
		extra_padding = int(padded_info, 2)

		padded_encoded_text = padded_encoded_text[8:] 
		encoded_text = padded_encoded_text[:-1*extra_padding]

		return encoded_text

	def decode_text(self, encoded_text):
		current_code = ""
		decoded_text = ""

		for bit in encoded_text:
			current_code += bit
			if(current_code in self.reverse_mapping):
				character = self.reverse_mapping[current_code]
				decoded_text += character
				current_code = ""

		return decoded_text
	def decompress(self, input_path):
		filename, file_extension = os.path.splitext(self.path)
		output_path = filename + "_decompressed" + ".txt"

		with open(input_path, 'rb') as file, open(output_path, 'w') as output:
			bit_string = ""

			byte = file.read(1)
			while(len(byte) > 0):
				byte = ord(byte)
				bits = bin(byte)[2:].rjust(8, '0')
				bit_string += bits
				byte = file.read(1)

			encoded_text = self.remove_padding(bit_string)

			decompressed_text = self.decode_text(encoded_text)
			
			output.write(decompressed_text)

		print("Decompressed")
		return output_path

