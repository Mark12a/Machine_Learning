#coding=utf-8
# @Author: Wei Ni
# @Time: 2019-07-23


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# parameter settings
learning_rate = 0.00002
num_iter = 5000000


def read_test_csv(file_path):
	test_data = pd.read_csv(file_path, header=None, sep=',')
	test_data.replace(to_replace="NR", value=0, inplace=True) #使用 value 值代替 DataFrame 中的 to_replace 值
	del test_data[0]
	del test_data[1]

	# test_data=pd.DataFrame(test_data, dtype=np.float)
	test_data = test_data.values
	test_data_cnt = 0
	while (test_data_cnt+17 <= test_data.shape[0]-1):
		test_data_temp = np.array(test_data[test_data_cnt:test_data_cnt+18, :])
		if test_data_cnt == 0:
			test_data_process = test_data_temp
		else:
			test_data_process = np.c_[test_data_process, test_data_temp]
		test_data_cnt += 18
	return test_data_process


def te_data_process(data):
	cnt = 0
	data = pd.DataFrame(data, dtype = np.float)
	# data = pd.DataFrame(data, dtype = np.float)
	data = data.values
	while (cnt+8 <= data.shape[1]-1):
		data_temp = data[:, cnt:cnt+9]
		data_temp = np.append(data_temp, np.float(1))
		data_temp = data_temp.reshape((1, -1))
		if cnt == 0:
			data_process = data_temp
			data_process = data_process.reshape((1, -1))
		else:
			data_process = np.r_[data_process, data_temp]
		cnt += 9
	min_max_scaler = preprocessing.MinMaxScaler()
	data_process = min_max_scaler.fit_transform(data_process)
	# data_process = preprocessing.StandardScaler().fit_transform(data_process)
	return data_process


def te_data_process_sk(data):
	cnt = 0
	# data = pd.DataFrame(data, dtype = np.float)
	data = data.values
	while (cnt+8 <= data.shape[1]-1):
		data_temp = data[:, cnt:cnt+9]
		data_temp = data_temp.reshape((1, -1))
		if cnt == 0:
			data_process = data_temp
			data_process = data_process.reshape((1, -1))
		else:
			data_process = np.r_[data_process, data_temp]
		cnt += 9

	return data_process


def read_train_csv(file_path):
	# train_data = np.genfromtxt(file_path, delimiter=',', skip_header=True, encoding="Big5")
	train_data = pd.read_csv(file_path, header=None, sep=',', encoding='Big5')
	train_data.replace(to_replace="NR", value=0, inplace=True) #使用 value 值代替 DataFrame 中的 to_replace 值
	# train_data.describe()
	data_label = train_data.iloc[1:19, 2:3]
	# print('data_label = {}'.format(data_label))
	data_label = np.array(data_label).reshape(18).tolist()

	train_data.drop([0], inplace = True) # Delete the First index
	del train_data[0] # Delete the First colomn
	# del train_data[0]
	del train_data[1]
	del train_data[2]

	# new_columns = [i for i in range(25)]
	# train_data.columns = new_columns
	# print('train data = {}'.format(train_data))
	# print(train_data.iloc[0:1])

	# print(train_data)
	# train_data = train_data.values
	# try:
	# 	train_data = np.delete(train_data, 0, axis=0)
	# except (ValueError):
	# 	raise('cannot reshape array of size')
	
	# train_data = pd.DataFrame(train_data)
	# print(train_data.shape[1])
	train_data.columns = [i for i in range(train_data.shape[1])]
	# train_data.replace(to_replace='\n', value='', inplace=True)
	train_data=pd.DataFrame(train_data, dtype=np.float)
	# train_data.index = data_label

	train_data = train_data.values
	# pm2_goal = 
	# train_data = scale(train_data)
	train_data=pd.DataFrame(train_data, dtype=np.float)
	# train_data.reset_index(drop=True)
	# train_data = pd.DataFrame(train_data, index = [i for i in range(train_data.shape[0])], columns = [i for i in range(train_data.shape[0])])
	# print(train_data)

	train_data_process = None
	process_cnt = 1
	while (process_cnt+17 < len(train_data)):
		train_data_temp = None
		train_data_temp = np.array(train_data.iloc[process_cnt-1:process_cnt+17])
		if process_cnt == 1:
			train_data_process = train_data_temp
		else:
			train_data_process = np.c_[train_data_process, train_data_temp]
		process_cnt += 18
	
	train_data_process = pd.DataFrame(train_data_process, dtype=np.float)
	train_data_process.index = data_label

	pm2_goal = train_data_process.loc['PM2.5']
	# train_data_process = train_data_process.values
	# train_data_process = scale(train_data_process)
	# train_data_process = pd.DataFrame(train_data_process, dtype=np.float)
	# train_data_process.index = data_label

	'''
	# print(train_data)
	# print(train_data.shape)
	train_data = np.reshape(train_data, (18, -1))
	# print(train_data)
	# print(train_data.shape)

	train_data = pd.DataFrame(train_data, index = data_label, columns = [i for i in range(train_data.shape[1])])
	train_data=pd.DataFrame(train_data,dtype=np.float)
	# print(type(train_data.iloc[10,5]))
	# print(train_data.iloc[10,5])
	# 
	# train_data.columns = [i for i in range(train_data.shape[1])]
	# print(train_data)
	'''
	return train_data_process, pm2_goal


def tr_data_process(data, pm2_goal):
	data = pd.DataFrame(data, dtype = np.float)
	slide_window = 10
	mod_index = [i for i in range(slide_window-1)]
	del_index = [i for i in range(slide_window-1)]
	cnt = slide_window - 1
	pm2_goal = pm2_goal.values
	data_np = data.values
	data_process = None
	while (cnt <= data_np.shape[1]-1):
		data_temp = data_np[:, cnt-slide_window+1:cnt]
		data_temp = np.append(data_temp, np.float(1))
		if cnt == slide_window-1:
			data_process = data_temp
			data_process = data_process.reshape((1, -1))
		else:
			data_temp = data_temp.reshape((1, -1))
			data_process = np.r_[data_process, data_temp]
		cnt += 1
		if cnt%480 in mod_index:
			for i in range(slide_window-1):
				del_index.append(cnt+i)
			cnt = cnt + slide_window - 1
	min_max_scaler = preprocessing.MinMaxScaler()
	data_process = min_max_scaler.fit_transform(data_process)
	# data_process = preprocessing.StandardScaler().fit_transform(data_process)

	pm2_goal = np.delete(pm2_goal, del_index)
	return data_process, pm2_goal


def tr_data_process_sk(data, pm2_goal):
	data = pd.DataFrame(data, dtype = np.float)
	slide_window = 10
	mod_index = [i for i in range(slide_window-1)]
	del_index = [i for i in range(slide_window-1)]
	cnt = slide_window - 1
	pm2_goal = pm2_goal.values
	data_np = data.values
	data_process = None
	while (cnt <= data_np.shape[1]-1):
		data_temp = data_np[:, cnt-slide_window+1:cnt]
		if cnt == slide_window-1:
			data_process = data_temp
			data_process = data_process.reshape((1, -1))
		else:
			data_temp = data_temp.reshape((1, -1))
			data_process = np.r_[data_process, data_temp]
		cnt += 1
		if cnt%480 in mod_index:
			for i in range(slide_window-1):
				del_index.append(cnt+i)
			cnt = cnt + slide_window - 1
	pm2_goal = np.delete(pm2_goal, del_index)
	return data_process, pm2_goal


def liner_regression(x_train, y_train):
	Weight=np.ones(shape=(1,x_train.shape[1]))
	iter_cnt = 0
	loss_pre = 1000000
	for num in range(num_iter):
		WXPlusB = np.dot(x_train, Weight.T).reshape((1,-1))
		loss = np.linalg.norm(y_train-WXPlusB)/y_train.shape[0]
		if (loss-loss_pre) > 0:
			print("loss is raising")
			break
		loss_pre = loss
		w_gradient = -(2.0/x_train.shape[0]) * np.dot((y_train-WXPlusB), x_train)
		Weight = Weight - learning_rate * w_gradient
		if num%100000 == 0:
			print('{}th iter, loss is {}'.format(iter_cnt,loss))
		iter_cnt += 1
	print("Final loss : {}".format(loss))
	return Weight


def verify_res(weight, x_verify, y_verify):
	y_weight = np.dot(weight, x_verify.T)
	y_diff = y_weight-y_verify
	diff = np.linalg.norm(y_diff)/x_verify.shape[0]
	print("Result diff on Verify data is {}".format(diff))



if __name__ == "__main__":
	# my-method
	print("Learning-Rate is {},    num_iter is {}".format(learning_rate, num_iter))
	train_file_path = "/Users/mark/Desktop/Liner_Regression/train.csv"
	test_file_path = "/Users/mark/Desktop/Liner_Regression/test.csv"
	train_data, pm2_goal = read_train_csv(train_file_path)
	X_train, Y_train = tr_data_process(train_data, pm2_goal)
	x_train, x_verify, y_train, y_verify = train_test_split(X_train, Y_train)
	Weight = liner_regression(x_train, y_train)
	verify_res(Weight, x_verify, y_verify)
	test_data = read_test_csv(test_file_path)
	X_test = te_data_process(test_data)
	Y_test = np.dot(X_test, Weight.T)
	id_index = np.array([str('id_'+str(i)) for i in range(Y_test.shape[0])]).reshape((1,-1))
	Y_test = np.r_[id_index, Y_test.T].T
	Y_test = pd.DataFrame(Y_test, dtype = str)
	Y_test.to_csv("/Users/mark/Desktop/Test_res.csv", header=None)

	# sklearn-method
	# train_file_path = "/Users/mark/Desktop/Liner_Regression/train.csv"
	# test_file_path = "/Users/mark/Desktop/Liner_Regression/test.csv"
	# train_data, pm2_goal = read_train_csv(train_file_path)
	# X_train, Y_train = tr_data_process_sk(train_data, pm2_goal)
	# model2 = LinearRegression(fit_intercept=True, normalize=True)
	# model2.fit(X_train, Y_train)
	# test_data = read_test_csv(test_file_path)
	# X_test = te_data_process_sk(test_data)
	# Y_test = model2.predict(X_test).reshape((-1, 1))
	# id_index = np.array([str('id_'+str(i)) for i in range(Y_test.shape[0])]).reshape((1,-1))
	# Y_test = np.r_[id_index, Y_test.T].T
	# Y_test = pd.DataFrame(Y_test, dtype = str)
	# Y_test.to_csv("/Users/mark/Desktop/Test_res_sklearn1.csv", header=None)