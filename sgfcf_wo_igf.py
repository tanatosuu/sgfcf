import torch
from torch.autograd import Variable
import argparse
import numpy as np
import pandas as pd
import time
import gc

torch.cuda.set_device(1)



parser = argparse.ArgumentParser(description='Argument parser for the algorithm.')

parser.add_argument('--dataset', type=str, default='yelp', help='Dataset name')
parser.add_argument('--density', type=str, default='dense', help='Density setting: dense or sparse')
parser.add_argument('--k', type=int, default=100, help='The number of required features')
parser.add_argument('--beta', type=float, default=1.0, help='coef for the filter')
parser.add_argument('--alpha', type=float, default=0.0, help='param for G^2N')
parser.add_argument('--eps', type=float, default=0.5, help='param for G^2N')
parser.add_argument('--gamma', type=float, default=1.0, help='weight for non-low frequency')

args = parser.parse_args()
 
dataset = args.dataset
setting = args.density
k = args.k
beta = args.beta
alpha = args.alpha
eps = args.eps
gamma=args.gamma


if dataset=='yelp':
	user,item=25677,25815

elif dataset=='citeulike':
	user,item=5551,16980

elif dataset=='pinterest':
	user,item=37501,9836

elif dataset=='gowalla':
	user,item=29858,40981

else:
	raise NotImplementedError(f"Dataset Error!")


#hyperparameter setting
#for alpha and eps, using either is ok, default: alpha=0, eps=0.5
#Citeulike
#80%:
#k,beta,eps,gamma=1000,0.95,0.28,1.5
#20%: 
#k,beta,alpha,gamma=110,4.6,2.3,0.05
#Pinterest
#80%: 
#k,beta,eps,gamma=300,1.0,0.37,0.3
#20%: 
#k,beta,eps,gamma=60,2.0,0.37,0.07
#Yelp
#80%:
#k,beta,alpha,gamma=250, 1.0,19,0.5
#20: 
#k,beta,alpha,gamma=50, 2.0,5.3,0.025
#Gowalla
#80%: 
#k,beta,eps,gamma=650,1.3, 0.34,1.6
#20%: 
#k,beta,alpha,gamma=110,3.1,4.8,0.02~0.03



if setting=='dense':
	df_train=pd.read_csv("./"+dataset+r'/train.csv')
	df_test=pd.read_csv("./"+dataset+ r'/test.csv')
elif setting=='sparse':
	df_train=pd.read_csv("./"+dataset+r'/train_sparse.csv')
	df_test=pd.read_csv("./"+dataset+ r'/test_sparse.csv')
else:
	raise NotImplementedError(f"Dataset Error!")




def weight_func(value):
	return value**beta
	#return 1+value+value**2+value**3
	#return torch.exp(beta*value)
	#return 1/(1-beta*value)



freq_matrix=torch.zeros(user,item).cuda()
for row in df_train.itertuples():
	freq_matrix[row[1],row[2]]=1

test_data=[[] for i in range(user)]
for row in df_test.itertuples():
	test_data[row[1]].append(row[2])


D_u=1/(freq_matrix.sum(1)+alpha).pow(eps)
D_i=1/(freq_matrix.sum(0)+alpha).pow(eps)



D_u[D_u==float('inf')]=0
D_i[D_i==float('inf')]=0


norm_freq_matrix=D_u.unsqueeze(1)*freq_matrix*D_i








U,value,V=torch.svd_lowrank(norm_freq_matrix,q=k+200,niter=30)
#U,value,V=torch.svd(R)
value=value/value.max()


rate_matrix = (U[:,:k]*weight_func(value[:k])).mm((V[:,:k]*weight_func(value[:k])).t())

rate_matrix = rate_matrix/(rate_matrix.sum(1)).unsqueeze(1)




norm_freq_matrix = norm_freq_matrix.mm(norm_freq_matrix.t()).mm(norm_freq_matrix)

norm_freq_matrix = norm_freq_matrix/(norm_freq_matrix.sum(1)).unsqueeze(1)

rate_matrix = ( rate_matrix + gamma * norm_freq_matrix).sigmoid()


#rate_matrix = rate_matrix - (freq_matrix!=0)*1e3

rate_matrix = rate_matrix - freq_matrix*1e3

del U, V, value, D_u, D_i, freq_matrix,norm_freq_matrix
gc.collect()
torch.cuda.empty_cache()



def test():
	#calculate idcg@k(k={1,...,20})
	def cal_idcg(k=20):
		idcg_set=[0]
		scores=0.0
		for i in range(1,k+1):
			scores+=1/np.log2(1+i)
			idcg_set.append(scores)

		return idcg_set

	def cal_score(topn,now_user,trunc=20):
		dcg10,dcg20,hit10,hit20=0.0,0.0,0.0,0.0
		for k in range(trunc):
			max_item=topn[k]
			if test_data[now_user].count(max_item)!=0:
				if k<=10:
					dcg10+=1/np.log2(2+k)
					hit10+=1
				dcg20+=1/np.log2(2+k)
				hit20+=1

		return dcg10,dcg20,hit10,hit20



	#accuracy on test data
	ndcg10,ndcg20,recall10,recall20=0.0,0.0,0.0,0.0

	idcg_set=cal_idcg()
	for now_user in range(user):
		test_lens=len(test_data[now_user])

		#number of test items truncated at k
		all10=10 if(test_lens>10) else test_lens
		all20=20 if(test_lens>20) else test_lens
	
		#calculate dcg
		topn=rate_matrix[now_user].topk(20)[1]


		dcg10,dcg20,hit10,hit20=cal_score(topn,now_user)


		ndcg10+=(dcg10/idcg_set[all10])
		ndcg20+=(dcg20/idcg_set[all20])
		recall10+=(hit10/all10)
		recall20+=(hit20/all20)	


	

	ndcg10,ndcg20,recall10,recall20=round(ndcg10/user,4),round(ndcg20/user,4),round(recall10/user,4),round(recall20/user,4)
	print(ndcg10,ndcg20,recall10,recall20)




test()
