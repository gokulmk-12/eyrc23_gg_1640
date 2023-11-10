'''
*****************************************************************************************
*
*        		===============================================
*           		GeoGuide(GG) Theme (eYRC 2023-24)
*        		===============================================
*
*  This script is to implement Task 1A of GeoGuide(GG) Theme (eYRC 2023-24).
*  
*  This software is made available on an "AS IS WHERE IS BASIS".
*  Licensee/end user indemnifies and will keep e-Yantra indemnified from
*  any and all claim(s) that emanate from the use of the Software or 
*  breach of the terms of this agreement.
*
*****************************************************************************************
'''

# Team ID:			[ Team-ID ]
# Author List:		[ Names of team members worked on this file separated by Comma: Name1, Name2, ... ]
# Filename:			task_1a.py
# Functions:	    [`ideantify_features_and_targets`, `load_as_tensors`,
# 					 `model_loss_function`, `model_optimizer`, `model_number_of_epochs`, `training_function`,
# 					 `validation_functions` ]

####################### IMPORT MODULES #######################
import pandas
import torch
import numpy as np
###################### Additional Imports ####################
'''
You can import any additional modules that you require from 
torch, matplotlib or sklearn. 
You are NOT allowed to import any other libraries. It will 
cause errors while running the executable
'''
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader,TensorDataset
from torch.optim.lr_scheduler import ExponentialLR
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
##############################################################

################# ADD UTILITY FUNCTIONS HERE #################





##############################################################

def data_preprocessing(task_1a_dataframe):



	''' 
	Purpose:
	---
	This function will be used to load your csv dataset and preprocess it.
	Preprocessing involves cleaning the dataset by removing unwanted features,
	decision about what needs to be done with missing values etc. Note that 
	there are features in the csv file whose values are textual (eg: Industry, 
	Education Level etc)These features might be required for training the model
	but can not be given directly as strings for training. Hence this function 
	should return encoded dataframe in which all the textual features are 
	numerically labeled.
	
	Input Arguments:
	---
	`task_1a_dataframe`: [Dataframe]
						  Pandas dataframe read from the provided dataset 	
	
	Returns:
	---
	`encoded_dataframe` : [ Dataframe ]
						  Pandas dataframe that has all the features mapped to 
						  numbers starting from zero

	Example call:
	---
	encoded_dataframe = data_preprocessing(task_1a_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
	columns_to_encode = ['Gender','Education','EverBenched','City']


	label_encoders = {}
	for column in columns_to_encode:
		label_encoder = LabelEncoder()
		task_1a_dataframe[column] = label_encoder.fit_transform(task_1a_dataframe[column])
		label_encoders[column] = label_encoder
	encoded_dataframe=task_1a_dataframe
	
	columns_to_normalize=["JoiningYear","PaymentTier","Age","ExperienceInCurrentDomain","Education",'City']
	scaler = MinMaxScaler()
	encoded_dataframe[columns_to_normalize] = scaler.fit_transform(encoded_dataframe[columns_to_normalize])

	##########################################################

	return encoded_dataframe

def identify_features_and_targets(encoded_dataframe):
	'''
	Purpose:
	---
	The purpose of this function is to define the features and
	the required target labels. The function returns a python list
	in which the first item is the selected features and second 
	item is the target label

	Input Arguments:
	---
	`encoded_dataframe` : [ Dataframe ]
						Pandas dataframe that has all the features mapped to 
						numbers starting from zero
	
	Returns:
	---
	`features_and_targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label

	Example call:
	---
	features_and_targets = identify_features_and_targets(encoded_dataframe)
	'''

	#################	ADD YOUR CODE HERE	##################
	X = encoded_dataframe.drop(["LeaveOrNot"], axis=1)
	y = encoded_dataframe["LeaveOrNot"]
	feature_name = X.columns.tolist()

	# chi_selector = SelectKBest(chi2, k=5)
	# chi_selector.fit(X, y)
	# chi_support = chi_selector.get_support()
	# chi_feature = X.loc[:,chi_support].columns.tolist()
	rfe_selector = RFE(estimator=LogisticRegression(), n_features_to_select=4, step=50, verbose=5)
	rfe_selector.fit(X, y)
	rfe_support = rfe_selector.get_support()
	rfe_feature = X.loc[:,rfe_support].columns.tolist()
	# print(chi_feature)
	print(rfe_feature)
	features = encoded_dataframe[rfe_feature]
	target = encoded_dataframe['LeaveOrNot']
	features_and_targets=[features,target]
	##########################################################

	return features_and_targets


def load_as_tensors(features_and_targets):

	''' 
	Purpose:
	---
	This function aims at loading your data (both training and validation)
	as PyTorch tensors. Here you will have to split the dataset for training 
	and validation, and then load them as as tensors. 
	Training of the model requires iterating over the training tensors. 
	Hence the training sensors need to be converted to iterable dataset
	object.
	
	Input Arguments:
	---
	`features_and targets` : [ list ]
							python list in which the first item is the 
							selected features and second item is the target label
	
	Returns:
	---
	`tensors_and_iterable_training_data` : [ list ]
											Items:
											[0]: X_train_tensor: Training features loaded into Pytorch array
											[1]: X_test_tensor: Feature tensors in validation data
											[2]: y_train_tensor: Training labels as Pytorch tensor
											[3]: y_test_tensor: Target labels as tensor in validation data
											[4]: Iterable dataset object and iterating over it in 
												 batches, which are then fed into the model for processing

	Example call:
	---
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	'''

	#################	ADD YOUR CODE HERE	##################
	features=features_and_targets[0]
	target=features_and_targets[1]
	X_train, X_test, y_train, y_test = train_test_split(features.values,target.values, test_size=0.27, random_state=50)
	X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
	X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
	y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1,1)
	y_test_tensor=torch.tensor(y_test,dtype=torch.float32).view(-1,1)
	tensors_and_iterable_training_data=[X_train_tensor,X_test_tensor,y_train_tensor,y_test_tensor]
	##########################################################

	return tensors_and_iterable_training_data

class Salary_Predictor(nn.Module):
	'''
	Purpose:
	---
	The architecture and behavior of your neural network model will be
	defined within this class that inherits from nn.Module. Here you
	also need to specify how the input data is processed through the layers. 
	It defines the sequence of operations that transform the input data into 
	the predicted output. When an instance of this class is created and data
	is passed through it, the `forward` method is automatically called, and 
	the output is the prediction of the model based on the input data.
	
	Returns:
	---
	`predicted_output` : Predicted output for the given input data
	'''
	def __init__(self):
		super(Salary_Predictor, self).__init__()
		'''
		Define the type and number of layers
		'''
		#######	ADD YOUR CODE HERE	#######
		self.layer_1=nn.Linear(4,100)

		self.layer_2=nn.Linear(100,1)
		# self.layer_3=nn.Linear(100,1)
		# self.layer_4=nn.Linear(128,1)
		self.lrelu = nn.ReLU()
		self.sigmoid=nn.Sigmoid()
		# self.dropout = nn.Dropout(p=0.15)
		self.tanh=nn.Tanh()
		# self.batchnorm1 = nn.BatchNorm1d(120,eps=1e-05, momentum=0.1)
		# self.batchnorm2 = nn.BatchNorm1d(200,eps=1e-05, momentum=0.1)
		# self.batchnorm3 = nn.BatchNorm1d(64,eps=1e-05, momentum=0.1)
        		
		###################################	

	def forward(self, x):
		'''
		Define the activation functions
		'''
		#######	ADD YOUR CODE HERE	#######
		x = self.lrelu(self.layer_1(x))
		# x = self.batchnorm1(x)
		# x = self.dropout(x)
		x = self.lrelu(self.layer_2(x)) 
		# x = self.batchnorm2(x)
		# # x = self.dropout(x)
		# x = self.lrelu(self.layer_3(x))
		# x = self.lrelu(self.layer_4(x))
	
		# x = self.batchnorm3(x)
		# x = self.dropout(x)

		predicted_output = self.sigmoid(x)
        		
		###################################
		return predicted_output


def model_loss_function():
	'''
	Purpose:
	---
	To define the loss function for the model. Loss function measures 
	how well the predictions of a model match the actual target values 
	in training data.
	
	Input Arguments:
	---
	None

	Returns:
	---
	`loss_function`: This can be a pre-defined loss function in PyTorch
					or can be user-defined

	Example call:
	---
	loss_function = model_loss_function()
	'''
	#################	ADD YOUR CODE HERE	##################
	loss_function=nn.BCELoss()
	##########################################################
	
	return loss_function

def model_optimizer(model):
	'''
	Purpose:
	---
	To define the optimizer for the model. Optimizer is responsible 
	for updating the parameters (weights and biases) in a way that 
	minimizes the loss function.
	
	Input Arguments:
	---
	`model`: An object of the 'Salary_Predictor' class

	Returns:
	---
	`optimizer`: Pre-defined optimizer from Pytorch

	Example call:
	---
	optimizer = model_optimizer(model)
	'''
	#################	ADD YOUR CODE HERE	##################
	optimizer = optim.Adam(model.parameters(), lr=0.001)

	##########################################################

	return optimizer

def model_number_of_epochs():
	'''
	Purpose:
	---
	To define the number of epochs for training the model

	Input Arguments:
	---
	None

	Returns:
	---
	`number_of_epochs`: [integer value]

	Example call:
	---
	number_of_epochs = model_number_of_epochs()
	'''
	#################	ADD YOUR CODE HERE	##################
    
	number_of_epochs=250

	##########################################################

	return number_of_epochs
def binary_acc(y_pred, y_test):
    y_pred_tag = torch.round((y_pred))
    correct_results_sum = (y_pred_tag == y_test).sum().float()
    acc = correct_results_sum / y_test.shape[0]
    return acc

def training_function(model, number_of_epochs, tensors_and_iterable_training_data, loss_function, optimizer):
	'''
	Purpose:
	---
	All the required parameters for training are passed to this function.

	Input Arguments:
	---
	1. `model`: An object of the 'Salary_Predictor' class
	2. `number_of_epochs`: For training the model
	3. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors
	4. `loss_function`: Loss function defined for the model
	5. `optimizer`: Optimizer defined for the model

	Returns:
	---
	trained_model

	Example call:
	---
	trained_model = training_function(model, number_of_epochs, iterable_training_data, loss_function, optimizer)

	'''	
	#################	ADD YOUR CODE HERE	##################
	X_train_tensor = tensors_and_iterable_training_data[0]
	X_test_tensor = tensors_and_iterable_training_data[1]
	y_train_tensor = tensors_and_iterable_training_data[2]
	y_test_tensor = tensors_and_iterable_training_data[3]
	train_data = TensorDataset(X_train_tensor, y_train_tensor)
	test_data = TensorDataset(X_test_tensor, y_test_tensor)
	train_loader = DataLoader(train_data, shuffle=True, batch_size=8)
	test_loader = DataLoader(test_data, batch_size=1)
	EPOCHS = number_of_epochs
	best_train_loss = float('inf')
	no_improvement_count = 0
	for e in range(1, EPOCHS + 1):
		epoch_loss = 0
		epoch_acc = 0
	
	model.train()
	for e in range(1, EPOCHS+1):
		epoch_loss = 0
		epoch_acc = 0
		for X_batch, y_batch in train_loader:
			X_batch, y_batch = X_batch, y_batch
			optimizer.zero_grad()
			
			y_pred = model(X_batch)
			
			loss = loss_function(y_pred, y_batch)
			acc = binary_acc(y_pred, y_batch)
			loss.backward()
			optimizer.step()
			epoch_loss += loss.item()
			epoch_acc += acc.item()
		

		train_loss=epoch_loss/len(test_loader)
		
		print(f'Epoch [{e}/{EPOCHS}] | Training Loss: {epoch_loss / len(train_loader):.4f} | Training Accuracy: {epoch_acc / len(train_loader):.4f}')

        # Early stopping logic
		if train_loss < best_train_loss:
			best_test_loss = train_loss
			no_improvement_count = 0
		else:
			no_improvement_count += 1
		if no_improvement_count >=200:
			print(f'Early stopping after {e} epochs with no improvement in validation loss.')
			break
		

     # Return the trained model
	trained_model=model
	##########################################################

	return trained_model

def validation_function(trained_model, tensors_and_iterable_training_data):
	'''
	Purpose:
	---300
	This function will utilise the trained model to do predictions on the
	validation dataset. This will enable us to understand the accuracy of
	the model.

	Input Arguments:
	---
	1. `trained_model`: Returned from the training function
	2. `tensors_and_iterable_training_data`: list containing training and validation data tensors 
											 and iterable dataset object of training tensors

	Returns:
	---
	model_accuracy: Accuracy on the validation dataset

	Example call:
	---
	model_accuracy = validation_function(trained_model, tensors_and_iterable_training_data)

	'''	
	#################	ADD YOUR CODE HERE	##################
    
	X_test_tensor, y_test_tensor = tensors_and_iterable_training_data[1], tensors_and_iterable_training_data[3]
    
	trained_model.eval()
    
	y_pred = trained_model(X_test_tensor)
	# y_pred_tag = torch.zeros_like(y_pred)  # Initialize with zeros
	count = 0

	# Apply the threshold and set values to 1 where >= 0.5
	
	y_pred_tag = y_pred
	# print(y_pred_tag==y_test_tensor)
    
	y_test = y_test_tensor

	y_pred_tag = torch.round(y_pred)
	# for i in range(y_test.shape):
	# 	if y_test[i]==y_pred[i]:
	# 		count+=1
	# print(count)
	
	correct_results_sum = (y_pred_tag == y_test).sum().float()
    
	acc = correct_results_sum / y_test.shape[0]
    
	acc = torch.round(acc * 100)
    
	model_accuracy=acc
    
	##########################################################

	return model_accuracy

########################################################################
########################################################################
######### YOU ARE NOT ALLOWED TO MAKE CHANGES TO THIS FUNCTION #########	
'''
	Purpose:
	---
	The following is the main function combining all the functions
	mentioned above. Go through this function to understand the flow
	of the script

'''
if __name__ == "__main__":

	# reading the provided dataset csv file using pandas library and 
	# converting it to a pandas Dataframe
	task_1a_dataframe = pandas.read_csv('/home/rishwanth/EYRC/Task_1/Task_1A/task_1a_dataset.csv')

	# data preprocessing and obtaining encoded data
	encoded_dataframe = data_preprocessing(task_1a_dataframe)

	# selecting required features and targets
	features_and_targets = identify_features_and_targets(encoded_dataframe)

	# obtaining training and validation data tensors and the iterable
	# training data object
	tensors_and_iterable_training_data = load_as_tensors(features_and_targets)
	
	# model is an instance of the class that defines the architecture of the model
	model = Salary_Predictor()

	# obtaining loss function, optimizer and the number of training epochs
	loss_function = model_loss_function()
	optimizer = model_optimizer(model)
	number_of_epochs = model_number_of_epochs()

	# training the model
	trained_model = training_function(model, number_of_epochs, tensors_and_iterable_training_data, 
					loss_function, optimizer)

	# validating and obtaining accuracy
	model_accuracy = validation_function(trained_model,tensors_and_iterable_training_data)
	print(f"Accuracy on the test set = {model_accuracy}")

	X_train_tensor = tensors_and_iterable_training_data[0]
	x = X_train_tensor[0].unsqueeze(0)
	jitted_model = torch.jit.save(torch.jit.trace(model, (x)), "task_1a_trained_model.pth")