import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import scipy.io as sio




class ConvAE(nn.Module):
	def __init__(self, reg1=1.0, reg2=1.0):

		super(ConvAE, self).__init__()
		self.reg1 = reg1
		self.reg2 = reg2

		self.encode_l1 = nn.Conv2d(1, 5, kernel_size=5, stride=1)
		self.encode_l2 = nn.Conv2d(5, 3, kernel_size=3, stride=1)
		self.encode_l3 = nn.Conv2d(3, 3, kernel_size=3, stride=1)

		self.decode_l1 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1)
		self.decode_l2 = nn.ConvTranspose2d(3, 5, kernel_size=3, stride=1)
		self.decode_l3 = nn.ConvTranspose2d(5, 1, kernel_size=5, stride=1)


	
	def forward(self, X):

		latent = F.relu(self.encode_l3(F.relu(self.encode_l2(F.relu(self.encode_l1(X))))))
		output = F.relu(self.decode_l3(F.relu(self.decode_l2(F.relu(self.decode_l1(latent))))))
		return latent, output



def train(CAE, input_data, num_epochs, lr=1.0e-3):
	parameters = CAE.parameters()
	for param in CAE.parameters():
		if len(param.data.shape) > 1:
			param.data = nn.init.xavier_normal(torch.Tensor(param.data.shape))
	
	
	X = Variable(torch.Tensor(input_data))
	optim = torch.optim.Adam(CAE.parameters(), lr=lr)

	for epoch in range(num_epochs):

		latent, output = CAE(X)

		recon_loss = 0.5*(torch.sum((output - X)**2))
		optim.zero_grad()
		recon_loss.backward()
		optim.step()
		avg_recon_loss = recon_loss.data[0]/400

		if (epoch+1)%100 == 0:
			print("Iter : ",epoch+1)
			print ("Loss : %.8f" % avg_recon_loss)



if __name__ == "__main__":

	# load face images and labels
	data = sio.loadmat("../Deep-subspace-clustering-networks/Data/ORL_32x32.mat")
	img = data['fea']
	label = data['gnd']

	# face image clustering
	n_input = [32,32]
	kernel_size = [3,3,3]
	n_hidden = [3,3,5]

	img = np.reshape(img, [img.shape[0], 1, n_input[0], n_input[1]])

	all_subjects = 40

	reg1 = 1.0
	reg2 = 0.2

	CAE = ConvAE()

	train(CAE, img, 900)
	torch.save(CAE, "./AE_ORL.pt")

	#avg, med = testFace(img, label, CAE, all_subjects)

	#print('Mean : ', avg*100, 'Median : ', med*100)


