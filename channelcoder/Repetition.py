import numpy as np


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
					out_txt = str(node.char) +" : "+str(node.freq)+", "+out_txt
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
		plt.savefig('Test_dir/result_huffman.png')

class HuffmanCoding:
	def __init__(self, mapped_data,count,inp_data_unique_arr,inp_data_unique_arr_idx_arr,draw_huffmantree):
		self.mapped_data = mapped_data
		self.count = count
		self.inp_data_unique_arr = inp_data_unique_arr
		self.inp_data_unique_arr_idx_arr = inp_data_unique_arr_idx_arr
		self.draw_huffmantree = draw_huffmantree
		self.heap = []
		self.codes = {}
		self.max_code_len = 0
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
			if(not isinstance(other, self.HeapNode)):
				return False
			return self.freq == other.freq

	# functions for compression:
	def make_heap(self,):
		for i in range(self.count.size):
			node = self.HeapNode(self.inp_data_unique_arr[i],self.count[i])
			heapq.heappush(self.heap, node) # 최소힙으로 만들어짐. 기준은 freq

	def merge_nodes(self,):
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
			self.inp_data_unique_arr
			inp_char = root.char

			self.codes[inp_char] = current_code
			self.reverse_mapping[current_code] = inp_char
			root.code = current_code

			if self.max_code_len < len(current_code):
				self.max_code_len = len(current_code)
			return

		self.make_codes_helper(root.leftChild, current_code + "0")
		self.make_codes_helper(root.rightChild, current_code + "1")

	def make_codes(self):
		root = heapq.heappop(self.heap) #HeapNode 클래스
		current_code = ""
		self.make_codes_helper(root, current_code)

	def get_encoded_np(self,):
		code_arr = np.array([list(map(int, list(x))) + [2] * (self.max_code_len - len(x)) for a, x in sorted(self.codes.items(), key=lambda x: x[0])])
		code_num_arr = np.array([len(x) for a, x in sorted(self.codes.items(), key=lambda x: x[0])])

		mapped_data_to_code = code_arr[self.mapped_data].astype('uint8')
		mapped_data_to_code_bit_num = code_num_arr[self.mapped_data]
		#inp_class.u_array_to_code = np.array([inp_class.data_to_code_dict[x] for x in inp_class.u_array], dtype='uint8')
		#encoded_text_num = np.array([len(self.codes[x]) for x in self.inp_data_unique_arr])[inv].reshape(columned_inp.shape)
		#encoded_text = inp_class.u_array_to_code[inv]
		return mapped_data_to_code, code_arr,mapped_data_to_code_bit_num

	def compress(self,):
		self.make_heap()
		self.merge_nodes()  # 여기서 huffman coding에서 볼 수 있는  tree를 생성함. True를 통해 허프만 결과 저장가능
		self.make_codes()

		if self.draw_huffmantree:
			self.tree.drawTree()

		#encoded_np,encoded_num_np = self.get_encoded_np()
		encoded_np, code_arr,mapped_data_to_code_bit_num = self.get_encoded_np()

		return encoded_np,code_arr, mapped_data_to_code_bit_num


