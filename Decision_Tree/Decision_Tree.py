# coding = utf-8
# @ Author: Wei Ni
# @ Build Time: 2019-07-25
# @ Latest Update Time: 2019-07-26

import numpy as np
import pandas as pd
import math


def read_train_csv(file_path):
		df = open(file_path, 'r')
		# print(df.read())
		train_dataset = pd.read_csv(df,header=None)
		labelx = list(train_dataset.iloc[0])
		train_dataset.drop([0], inplace=True)
		new_index = range(len(train_dataset))
		train_dataset.index = new_index
		train_dataset.columns = labelx
		return train_dataset

class Node(object):
	def __init__(self, x=None, label=None, y=None, data=None):
		self.label = label # label:子节点分类依据的特征
		self.x = x         # x:特征
		self.child = []    # child:子节点
		self.y = y         # y:类标记（叶节点才有）
		self.data = data   # data:包含数据（叶节点才有


	def append(self, child):
		self.child.append(child)


	def predict(self, features):  # 预测类
		if self.y is not None:
			return self.y
		for ch in self.child:
			if ch.x == features[self.label]:
				return ch.predict(features)



class Decision_Tree_ID3():

	def __init__(self, threshold=0, file_path="", alpha=0):
		self.tree = Node()
		self.alpha = alpha
		self.file_path = file_path
		self.threshold = threshold


	def prob(self, dataset):
		dataset = dataset.T
		data_len = float(len(dataset))
		classes = set(dataset)
		class2val = dict(dataset.value_counts())
		for cla in classes:
			class2val[cla] = float(class2val[cla])/data_len
		return class2val


	def total_ent(self, dataset):
		p = self.prob(dataset)
		log_p = np.log2(list(p.values()))
		return -np.sum(np.multiply(list(p.values()), log_p))


	def attri_ent(self, dataset):
		attr_ent = []
		for col in dataset.columns[:-1]:
			attr = dataset[col]
			cluster = dataset[dataset.columns[-1]]
			attr_i2cluster = {i:[] for i in set(attr)}
			for i, d in enumerate(cluster):
				attr_i2cluster[attr.iloc[i]].append(d)
			attr_ent.append(sum([self.prob(dataset[col])[i] * self.total_ent(pd.Series(np.array(attr_i2cluster[i])))  for i in attr_i2cluster.keys()]))
		return attr_ent


	def gain_info(self, train_data):
		attri_label = train_data.columns
		ent_D = self.total_ent(train_data[attri_label[-1]])
		ent_attr = self.attri_ent(train_data)
		gain_attr = list(np.array(ent_D) - np.array(ent_attr))
		gain_max = max(gain_attr)
		gain_id2ent = dict(enumerate(gain_attr))
		gain_id2ent = dict(zip(gain_id2ent.values(), gain_id2ent.keys()))
		gain_max_id = gain_id2ent[gain_max]
		return gain_max_id, gain_max


	def train(self, train_data, node):
		# Dataset is empty
		if train_data.empty:
			print("The data is empty")
			return

		cluster_data = train_data.columns[-1]

		# Labels are the same
		if len(train_data[cluster_data].value_counts()) == 1:
			node.data = train_data[cluster_data]
			node.y = train_data[cluster_data][0]
			print("All labels are the same")
			return
		# No atrribute
		elif len(train_data.columns[:-1]) == 0:
			node.data = train_data[cluster_data]
			node.y = train_data[cluster_data].value_counts().index[0]
			print("No attributes ")
			return
		# All attributes are the same
		else:
			flag = True
			for attri in train_data.columns[:-1]:
				if (len(train_data[attri].value_counts()) == 1):
					flag = True
				else:
					flag = False
					break
			if flag:
				print("All attributes are the same ")
				return

		attri_data = train_data[:-1]
		gain_max_id, gain_max = self.gain_info(train_data)
		# 任何一个信息增益小于0，返回单节点树T，标记为样本类别数最多的类别。
		if gain_max < 0:
			node.data = cluster_data
			node.y = cluster_data.value_counts.index[0]
			return
		# 按特征的不同取值将对应的样本输出D分成不同的类别Di
		# 每个类别产生一个子节点。对应特征值为Agi。返回增加了节点的数T
		value_cnt = train_data[train_data.columns[gain_max_id]].value_counts()
		for gainmax_i in value_cnt.index:
			node.label = gain_max_id
			child = Node(gainmax_i)
			node.append(child)
			new_datasets = pd.DataFrame([list(i) for i in train_data.values if i[gain_max_id]==gainmax_i], columns=train_data.columns)
			self.train(new_datasets, child)
	
	def fit(self, train_data):
		self.train(train_data, self.tree)


def printnode(node, depth=0):  # 打印树所有节点
	if node.label is None:
		print(depth, (node.label, node.x, node.y, len(node.data)))
	else:
		print(depth, (node.label, node.x))
		for c in node.child:
			printnode(c, depth+1)
	# self.label = label # label:子节点分类依据的特征
	# self.x = x         # x:特征
	# self.child = []    # child:子节点
	# self.y = y         # y:类标记（叶节点才有）
	# self.data = data   # data:包含数据（叶节点才有


			
if __name__ == "__main__":
	train_path = "/Users/mark/Desktop/Machine_Learning/Decision_Tree/car_train.csv"
	test_path = "/Users/mark/Desktop/Machine_Learning/Decision_Tree/car_test.csv"
	train_dataset = read_train_csv(train_path)
	dt_ID3 = Decision_Tree_ID3()
	dt_ID3.fit(train_dataset)
	printnode(dt_ID3.tree)
	print(dt_ID3.tree.predict(['vhigh','vhigh','4','4','med','med']))
	# X_train, Y_train = read_train_csv(train_path)