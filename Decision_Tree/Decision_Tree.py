# coding = utf-8
# @ Author: Wei Ni
# @ Build Time: 2019-07-25
# @ Latest Update Time: 2019-07-26

import numpy as np
import pandas as pd
import math


class Node():
	def __init__(self, x=None, label=None, y=None, data=None):
		self.label = label # label:子节点分类依据的特征
		self.x = x         # x:特征
		self.child = []    # child:子节点
		self.y = y         # y:类标记（叶节点才有）
		self.data = data   # data:包含数据（叶节点才有


	def append(self, child):
		self.child.append(child)


	def predict_label(self):
		if self.y is not None:
			return self.y
		else:
			pass



class Decision_Tree_ID3():

	def __init__(self, threshold=0, file_path="", alpha=0):
		self.dataset = dataset
		self.tree = Node()
		self.alpha = alpha
		self.file_path = file_path
		self.threshold = threshold


	def read_train_csv(self, file_path):
		train_dataset = pd.read_csv(file_path, header=None, sep=',')
		train_dataset.drop([0], inplace=True)
		train_dataset = train_dataset.values
		x_train = train_dataset[:, :-1]
		y_train = train_dataset[:, train_dataset.shape[1]-1]


	def prob(self, dataset):
		dataset = dataset.T
		data_len = float(len(dataset))
		classes = set(dataset)
		class2val = dict(dataset.value_counts())
		probability = []
		for cla in classes:
			probability.append(float(class2val[cla])/data_len)
		return probability

	def attri_ent(self, dataset):
		pass


	def total_ent(self, dataset):
		p = prob(dataset)
		log_p = np.log2(p)
		return -np.sum(np.multiply(p, log_p))


	def attri_ent(self, dataset):
		attr_ent = []
		for col in dataset.columns[:-1]:
			attr = dataset[col]
			cluster = dataset[dataset.columns[-1]]
			attr_i2cluster = {i:[] for i in set(attr)}
			for i, d in enumerate(cluster):
				attr_i2cluster[attr.iloc[i]].append(d)
			attr_ent.append(sum([prob(attr_i2cluster[i])[i]*total_ent(attr_i2cluster[i])  for i in attr_i2cluster.keys()]))


	def gain_info(self, train_data):
		attri_label = train_data.columns
		ent_D = total_ent(train_data[attri_label[-1]])
		ent_attr = attri_ent(dataset)


	def train(self, train_data, node):
		# Dataset is empty
		if train_data.empty:
			print("The data is empty")
			return

		cluster_data = train_data.columns[-1]

		# Labels are the same
		if (len(train_data[cluster_data].value_counts()) == 1):
			print("All labels are the same")
			return
		# No atrribute
		elif (len(train_data.columns[:-1]) == 0):
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
		gain_data, gain_max = self.gain_info(train_data)
		# 任何一个信息增益小于0，返回单节点树T，标记为样本类别数最多的类别。
		if gain_max < 0:
			node.data = cluster_data
			node.y = cluster_data.value_counts.index[0]
			return
		# 按特征的不同取值将对应的样本输出D分成不同的类别Di
		# 每个类别产生一个子节点。对应特征值为Agi。返回增加了节点的数T
		value_cnt = train_data[train_data.columns[gain_max]].value_counts()
		for gainmax_i in value_cnt.index:
			node.label = gain_max
			child = Node(gainmax_i)
			node.append(child)
			new_datasets = pd.DataFrame([list(i) for i in train_data.values if i[gain_max]==gainmax_i], columns=train_data.columns)
			self.train(new_datasets, child)
	
	def fit(self, train_data):
		self.train(train_data, self.tree)



if __name__ == "__main__":
	train_path = "/Users/mark/Desktop/Decision_Tree/car_train.csv"
	test_path = "/Users/mark/Desktop/Decision_Tree/car_test.csv"
	# X_train, Y_train = read_train_csv(train_path)
