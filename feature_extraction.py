#!/net/ugrads/gnarang/pvt/mypyth/bin/python

import numpy as np
from re import search
import getopt
import sys

import collections

args=sys.argv

#num_VFI=4

VFI=[args[1], args[2], args[3], args[4]]
#print(VFI)
#print(float(VFI[0]))

epoch = args[5]
e = int(epoch)
#print(epoch)

folder_out=args[6]

data_folder=args[7]

##############################

#use clustering.txt from folder
cl=np.loadtxt(data_folder+'clustering.txt', dtype=int)
#print(cl) # cl is array

#voltages
voltage = ['1.0','0.95','0.9','0.85','0.8','0.75','0.7','0.65']
volt_strings = ['10','095','09','085','08','075','07','065']

util_VFI = np.zeros((4,64)) # 4 VFI, 64 core , only fill those places where cl[j]===i, keep count to find average utilization of each VFI
traffic_VFI = np.zeros((4,64))
pwr_VFI = np.zeros((4,64))
simsec_VFI = np.zeros((4,64))

count_avg = np.zeros((4,1)) # 4 VFI (for avg ipc)
count_avg_tr = np.zeros((4,1)) # 4 VFI (for avg traffic)

features=np.zeros((4,4))

#ipc_avg=np.array((4,1))
#ipc_max=np.array((4,1))

################# reading stdout1.txt ##############
src_count=0
src_list=[]

#with open(data_folder+'stdout1.txt') as f1:

with open(data_folder+'../fft/stdout1.txt') as f1: #fixing to fft/stdout because it is same and all workload dont have this file
	line = f1.readlines()
	#print(line)
	for s in line:
		if "src=" in s: # if substring 'src=X' matches, break those lines
			#src_count+=1
			split1=s.split() #split by space
			split2=split1[0].split('=') # split by '='
			src_list.append(split2[1])
#print(src_list)
#print(src_count)


#src_dict = {i:src_list.count(i) for i in src_list}
#print(src_dict)

#################

for i in range(4): # for 4 VFI
	volt=VFI[i]
	v_i = voltage.index(volt)
	v = volt_strings[v_i]

	############ UTILIZATION ###############
	util=np.loadtxt(data_folder+'utilization'+v+'.txt')
	util_64_epoch_N = util[e]

	for j in range(len(cl)):#j 0->63
		if cl[j]==i:		
			util_VFI[i,j] = util_64_epoch_N[j]
			count_avg[i]+=1	#need to refresh this count
	#print("count_avg",count_avg)		
	#print("util",util_VFI[i])
	#print("avg",np.sum(util_VFI[i,:])/count_avg[i])

	ipc_avg=np.sum(util_VFI[i,:])/count_avg[i]
	ipc_max=np.max(util_VFI[i,:])


	################ TRAFFIC #################
	traffic_64_epoch_N=np.zeros((1,292), dtype=int) # to take care of 1 missing 0 in epoch 322 in traffic database

	f2=open(data_folder+'traffic'+v+'.txt')
	line=f2.readlines()
	#print(line[e])
	line = [x.strip() for x in line] # strip newline from every epoch
	l=line[e].split()	#split every epoch into list to remove whitespaces between elements
	#print("len here list",len(l))
	for m in range(len(l)):
		#print(l[m])
		traffic_64_epoch_N[0,m] = l[m]
	#print(traffic_64_epoch_N)
	#print("len here = = ",traffic_64_epoch_N.shape) # (1,292)

	'''
	c_prev=0
	t=0
	traffic_core= []
	
	print("DICT",src_dict.get("0"))
	for i in range(64):
		c= src_dict['i']
		print(c)
		for j in range(c):
			t +=  traffic_64_epoch_N[j+c_prev]
			c_prev = c
		traffic_core.append(t)
	print(len(traffic_core))
	'''
	
	# create 2d array 64x64
	# in row 0 , add traffic elements from index given by src_list elements
	# once created, sum rows of 2d array to output 1d array of ((64,1))
	# use 1d array to create per-VFI traffic max and avg


	# 1. compute 2d array
	traffic_2d = np.zeros((64,64))
	col=0
	#prev_row=0
	
	for j in range(len(src_list)):		#len(src_list)
		row=int(src_list[j])
		if j==(len(src_list)-1):
			next_row=row
		else:
			next_row=int(src_list[j+1])

		#print(traffic_64_epoch_N[j])
		#print(row)
		#print(col)
		#print('-----')

		traffic_2d[row,col]= traffic_64_epoch_N[0,j]
		if (next_row == row):
			col+=1
		else:
			col=0
		

	#print(traffic_2d[0:7,0:10])

	
	### 2. make 1-d array ###
	traffic_per_core=np.zeros((64,1))

	for k in range(64):
		traffic_per_core[k]=np.sum(traffic_2d[k])/64 #just a normalization factor
	#print(traffic_per_core)

	#### 3. per-VFI data ###
	
	for j in range(len(cl)):#j 0->63
		if cl[j]==i:		
			traffic_VFI[i,j] = traffic_per_core[j]
			count_avg_tr[i] += 1

	# 4. avg and max traffic
	tr_avg=np.sum(traffic_VFI[i,:])/count_avg_tr[i]/100
	tr_max=np.max(traffic_VFI[i,:])/100 # normalize	



	###############################
	
	# combine all VFI features and save
	features[i,:]=ipc_avg, ipc_max, tr_avg, tr_max

	## POWER ##

	pwr=np.loadtxt(data_folder+'power'+v+'.txt')
	pwr_64_epoch_N = pwr[e]

	for j in range(len(cl)):#j 0->63
		if cl[j]==i:		
			pwr_VFI[i,j] = pwr_64_epoch_N[j]


	## exec time ##
	# max(simsec of cores in VFI-i)* power of that VFI = edp
	simsec=np.loadtxt(data_folder+'simSec'+v+'.txt')
	simsec_64_epoch_N = simsec[e]

	for j in range(len(cl)):#j 0->63
		if cl[j]==i:		
			simsec_VFI[i,j] = simsec_64_epoch_N[j]


# out of VFI loop

#################
#print(features)
np.savetxt(folder_out+'features'+str(int(epoch)+1)+'.txt', features) # now you should see all featuresN.txt wrt epochs

#################
#total power = sum of all cores power of VFI-i

power_per_epoch=np.sum(pwr_VFI)
#print(power_per_epoch)
np.savetxt(folder_out+'power'+str(int(epoch)+1)+'.txt', [power_per_epoch]) # power each epoch #0-d array not supported -> make list []
#################
# max(simsec)

simsec_per_epoch = np.max(simsec_VFI)
#print(simsec_per_epoch)
np.savetxt(folder_out+'simsec'+str(int(epoch)+1)+'.txt', [simsec_per_epoch])



# end of program
#######################