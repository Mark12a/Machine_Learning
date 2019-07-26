# coding = utf-8
# @ Author: Wei Ni
# @ Build Time: 2019-07-25
# @ Latest Update Time: 2019-07-26

import numpy as np
import pandas as pd


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


	def gain_info(self, attri_data, cluster_data):
		pass


	def train(self, train_data ):
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
		elif ():
			pass

		attri_data = train_data[:-1]
		gain_data, gain_max = self.gain_info(attri_data, cluster_data)
		



if __name__ == "__main__":
	train_path = "/Users/mark/Desktop/Decision_Tree/car_train.csv"
	test_path = "/Users/mark/Desktop/Decision_Tree/car_test.csv"
	# X_train, Y_train = read_train_csv(train_path)
