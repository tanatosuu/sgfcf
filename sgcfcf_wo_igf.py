import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc


dataset='./yelp'

user,item=25677,25815
#user,item=6040,3952
#user,item=5551,16980
#user,item=37501,9836
#user,item=29858,40981


#hyperparameter setting
#Citeulike
#80%: k=1000, beta=0.95, eps=0.28-0.29, gamma=1.5
#20%: k=140, beta=4.6, alpha=6.3, gamma=0.05
#Pinterest
#80%: k=300, beta=1.0, eps=0.37, gamma=0.3
#20%: k=60, beta=2.0, alpha=6.8, gamma=0.07
#Yelp
#80%: k=250, beta=1.0, alpha=19, gamma=0.5
#20: k=40, beta=2.0, alpha=5.3, gamma=0.025
#Gowalla
#80%: k=700, beta=1.3, alpha=15, gamma=1.6
#20%: k=110, beta=3.1, alpha=4.8, gamma=0.02~0.03



beta, k=2.0, 50

#using either is okay, default: alpha=0, eps=0.5
alpha,eps=5.3,0.5
gamma=0.025


df_train=pd.read_csv(dataset+r'/train_sparse.csv')
df_test=pd.read_csv(dataset+ r'/test_sparse.csv')

def weight_func(value):
	return value**beta
	#return 0.25+0.75*value+2.25*value**2+1.75*value**3
	#return 0.0625-0.25*value+0.9375*value**2+1.75*value**3
	#return 1+value+value**2+value**3
	#return torch.exp(beta*value)
	#return 1/(1-0.96*value)

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