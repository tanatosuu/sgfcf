import torch
from torch.autograd import Variable
import numpy as np
import pandas as pd
import time
import gc

torch.cuda.set_device(1)

dataset='./yelp'

user,item=25677,25815
#user,item=6040,3952
#user,item=5551,16980
#user,item=37501,9836
#user,item=29858,40981


#hyperparamter setting for [beta_1, beta_2]
#Citeulike
#80%: [0.9,1.1]
#20%: [4.3,5.1]
#Pinterest
#80%: almost does not bring improvemnt
#20%: [2.0,2.4]
#Yelp
#80%: [1.0,1.3]
#20: [2.0,5.0]
#Gowalla
#80%: [0.3,2.5]
#20%: [2.5,6.5]

k=50
beta_1,beta_2=2.0,5.0

#using either is okay, default: alpha=0, eps=0.5
alpha,eps=5.3,0.5
gamma=0.025


df_train=pd.read_csv(dataset+r'/train_sparse.csv')
df_test=pd.read_csv(dataset+ r'/test_sparse.csv')


freq_matrix=torch.zeros(user,item).cuda()
for row in df_train.itertuples():
	freq_matrix[row[1],row[2]]=1

test_data=[[] for i in range(user)]
for row in df_test.itertuples():
	test_data[row[1]].append(row[2])



def individual_weight(value,homo_ratio):
	y_min,y_max=beta_1,beta_2
	x_min,x_max=homo_ratio.min(),homo_ratio.max()
	homo_weight=(y_max-y_min)/(x_max - x_min)*homo_ratio+(x_max*y_min - y_max*x_min)/(x_max - x_min)
	homo_weight=homo_weight.unsqueeze(1)


	#return torch.exp(homo_weight*value)
	return value.pow(homo_weight)




homo_ratio_user,homo_ratio_item=[],[]

train_data=[[] for i in range(user)]
train_data_item=[[] for i in range(item)]
for row in df_train.itertuples():
	train_data[row[1]].append(row[2])
	train_data_item[row[2]].append(row[1])

for u in range(user):
	if len(train_data[u])>1:
		inter_items=freq_matrix[:,train_data[u]].t()
		inter_items[:,u]=0
		connect_matrix=inter_items.mm(inter_items.t())
		
		#connect_matrix=connect_matrix+connect_matrix.mm(connect_matrix)+connect_matrix.mm(connect_matrix).mm(connect_matrix)

		size=inter_items.shape[0]
		#homo_ratio
		ratio_u=((connect_matrix!=0).sum().item()-(connect_matrix.diag()!=0).sum().item())/(size*(size-1))
		homo_ratio_user.append(ratio_u)
	else:
		homo_ratio_user.append(0)


for i in range(item):
	if len(train_data_item[i])>1:
		inter_users=freq_matrix[train_data_item[i]]
		inter_users[:,i]=0
		connect_matrix=inter_users.mm(inter_users.t())

		#connect_matrix=connect_matrix+connect_matrix.mm(connect_matrix)+connect_matrix.mm(connect_matrix).mm(connect_matrix)

		size=inter_users.shape[0]
		#homo_ratio
		ratio_i=((connect_matrix!=0).sum().item()-(connect_matrix.diag()!=0).sum().item())/(size*(size-1))
		homo_ratio_item.append(ratio_i)
	else:
		homo_ratio_item.append(0)

homo_ratio_user=torch.Tensor(homo_ratio_user).cuda()
homo_ratio_item=torch.Tensor(homo_ratio_item).cuda()


D_u=1/(freq_matrix.sum(1)+alpha).pow(eps)
D_i=1/(freq_matrix.sum(0)+alpha).pow(eps)




D_u[D_u==float('inf')]=0
D_i[D_i==float('inf')]=0


norm_freq_matrix=D_u.unsqueeze(1)*freq_matrix*D_i






U,value,V=torch.svd_lowrank(norm_freq_matrix,q=k+200,niter=30)
#U,value,V=torch.svd(R)
value=value/value.max()

rate_matrix = (U[:,:k]*individual_weight(value[:k],homo_ratio_user)).mm((V[:,:k]*individual_weight(value[:k],homo_ratio_item)).t())

#rate_matrix=rate_matrix-rate_matrix.min(1)[0].unsqueeze(1)

del homo_ratio_user, homo_ratio_item
gc.collect()
torch.cuda.empty_cache()



rate_matrix = rate_matrix/(rate_matrix.sum(1).unsqueeze(1))


norm_freq_matrix = norm_freq_matrix.mm(norm_freq_matrix.t()).mm(norm_freq_matrix)

norm_freq_matrix = norm_freq_matrix/(norm_freq_matrix.sum(1).unsqueeze(1))


rate_matrix = ( rate_matrix + gamma * norm_freq_matrix).sigmoid()


rate_matrix = rate_matrix - freq_matrix*1000

del U, V, value, D_u, D_i, freq_matrix, norm_freq_matrix
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