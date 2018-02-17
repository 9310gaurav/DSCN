import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import scipy.io as sio
import scipy.sparse as sparse

from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.preprocessing import normalize
from munkres import Munkres


class ConvAE(nn.Module):
	def __init__(self, batch_size):

		super(ConvAE, self).__init__()
		self.batch_size = batch_size

		self.encode_l1 = nn.Conv2d(1, 5, kernel_size=5, stride=1)
		self.encode_l2 = nn.Conv2d(5, 3, kernel_size=3, stride=1)
		self.encode_l3 = nn.Conv2d(3, 3, kernel_size=3, stride=1)

		self.coeff = nn.init.xavier_normal(torch.Tensor(batch_size, batch_size))
		c = self.coeff.numpy()

		temp = ((sparse.rand(400, 400, density = 0.999)).toarray()) > 0
		c[temp] = 0
		print(np.square(c).sum())
		c = c - np.eye(batch_size, batch_size)*c
		print(np.square(c).sum())
		self.coeff = torch.Tensor(c)

	#	self.coeff = torch.Tensor(np.eye(batch_size,batch_size))
		self.coeff = Variable(self.coeff, requires_grad = True)

		self.decode_l1 = nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1)
		self.decode_l2 = nn.ConvTranspose2d(3, 5, kernel_size=3, stride=1)
		self.decode_l3 = nn.ConvTranspose2d(5, 1, kernel_size=5, stride=1)


	
	def forward(self, X):

		latent = F.relu(self.encode_l3(F.relu(self.encode_l2(F.relu(self.encode_l1(X))))))
		latent_c = self.coeff.mm(latent.view(self.batch_size, -1))
		latent_c = latent_c.view(latent.size())

		output = F.relu(self.decode_l3(F.relu(self.decode_l2(F.relu(self.decode_l1(latent_c))))))
		return latent, latent_c, self.coeff, output



def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)
    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2  


def thrC(C, ro):
	if ro < 1:
		N = C.shape[1]
		Cp = np.zeros((N,N))
		S = np.abs(np.sort(-np.abs(C), axis=0))
		Ind = np.argsort(-np.abs(C), axis=0)
		for i in range(N):
			cL1 = np.sum(S[:,i]).astype(float)
			stop = False
			csum = 0
			t = 0
			while stop == False:
				csum = csum + S[t,i]
				if csum > ro*cL1:
					stop = True
					Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
				t = t + 1
	else:
		Cp = C

	return Cp


def post_proC(C, K, d, alpha):

	C = 0.5*(C + C.T)
	r = min(d*K + 1, C.shape[0] - 1)
	U, S, _ = svds(C, r , v0 = np.ones(C.shape[0]))
	U = U[:,::-1]
	S = np.sqrt(S[::-1])
	S = np.diag(S)
	U = U.dot(S)
	U = normalize(U, norm='l2', axis=1)
	Z= U.dot(U.T)
	Z = Z*(Z>0)
	L = np.abs(Z**alpha)
	L = L/L.max()
	L = 0.5*(L + L.T)
	spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed', assign_labels='discretize')
	spectral.fit(L)
	grp = spectral.fit_predict(L) + 1
	return grp, L


def err_rate(gt_s, s):
	c_x = best_map(gt_s, s)
	err_x = np.sum(gt_s[:] != c_x[:])
	missrate = err_x.astype(float) / (gt_s.shape[0])
	return missrate


def train(CAE, input_data, label, num_epochs, lr=1.0e-3, reg1=1.0, reg2=1.0, restore_path=""):


	alpha = 0.2
	face_10_subjs = np.array(input_data)
	face_10_subjs = face_10_subjs.astype(float)        
	label_10_subjs = np.array(label) 
	label_10_subjs = label_10_subjs - label_10_subjs.min() + 1
	label_10_subjs = np.squeeze(label_10_subjs)
	
	PreAE = torch.load(restore_path)
	i = 0

	params_preAE = dict([(name, param) for name,param in PreAE.named_parameters()])

	for name,param in CAE.named_parameters():

		if name in params_preAE:
			param_pre = params_preAE[name]
			param.data = param_pre.data



	X = Variable(torch.Tensor(input_data), requires_grad=False)
	optim = torch.optim.Adam(CAE.parameters(), lr=lr)

	for epoch in range(num_epochs):

		if epoch > 5000:
			optim = torch.optim.Adam(CAE.parameters(), lr=lr)

		latent, latent_c, coeff, output = CAE(X)

		#print(torch.sum(latent**2))

		#l2 reconstruction loss
		recon_loss = 0.5*(torch.sum((output - X)**2))

		#l2 regularization loss
		reg_loss = torch.sum(coeff**2)

		#expressiveness loss
		exp_loss = 0.5*torch.sum((latent - latent_c)**2)

		loss = recon_loss + reg1*reg_loss + reg2*exp_loss

		optim.zero_grad()
		loss.backward()
		optim.step()
		avg_loss = loss.data[0]/400

		if (epoch+1)%1 == 0 or epoch == 0:
			print("Iter : ",epoch+1)
			print ("Recon Loss : %.8f   Reg loss : %.8f   Exp loss : %.8f   Avg loss : %.8f" % (recon_loss/400, reg_loss, exp_loss, avg_loss))

			Coef = thrC(coeff.data.numpy(), alpha)
			y_x, _ = post_proC(Coef, label_10_subjs.max(), 3,1)
			missrate_x = err_rate(label_10_subjs, y_x)
			acc_x = 1 - missrate_x
			print "experiment: %d" % i, "our accuracy: %.4f" % acc_x






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

	batch_size = 400

	reg1 = 1.0
	reg2 = 0.2


	CAE = ConvAE(batch_size)

	train(CAE, img, label, 500, 1.0, 0.2, restore_path="./AE_ORL.pt")

	#avg, med = testFace(img, label, CAE, all_subjects)

	#print('Mean : ', avg*100, 'Median : ', med*100)