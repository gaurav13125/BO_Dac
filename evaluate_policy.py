#!/net/ugrads/gnarang/pvt/mypyth/bin/python

# 1. run gem5 for entire workload (say FFT) at nominal voltage to all 64 cores
# 2. using the info from this run, do clustering and divide 64 cores into 4 VFIs
#    gem5 script fs.py will update according to this clustering info
# 3. extract other parameters from stats.txt such as 
#    "sim_insts" - Number of instructions simulated
# 4. lets say you want to run for 5 epochs, take checkpoints using MOESI_hammer binary at every I=sim_insts/5
# 5. restore from respective checkpoints using MESI_TwoLevel binary with modified voltages values for each VFI
# 6. How to calculate avg and max IPC ?
#    say in VFI1, cores 1,5,7,10 are present.
#    system.cpu01.numCycles = gives cycle of each core, take the max of them  i.e. max_cycle
#    system.switch_cpus03.iq.issued_per_cycle::total =  instructions per cpu
#    avg IPC = sum of all instructions/ max_cycle or sum of 'committedInsts'/max_cycle
#    max IPC = max(iq.issued_per_cycle of all cores in respective VFI)
# 7. How to get Fij per VFI (traffic or injection rate) ?
#    -modify garnet to print it



from sklearn import tree
from random import shuffle
from sklearn import preprocessing
#from sklearn.ensemble import RandomForestRegressor
#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor, MLPClassifier
import copy
import numpy as np

import os
import subprocess

import time


#from keras.models import Sequential
#from keras.layers import Dense






#relu
def f_relu( X):
	return np.maximum(0, X)

#softmax
def f_softmax( X):
        expo = np.exp(X)
        expo_sum = np.sum(np.exp(X))
        return expo/expo_sum

    
# Predict Frequency
def f_pred_direct(features, regressor, num_inputs, num_neurons, num_actions):
        n_layer1_weights = num_inputs*num_neurons
        n_layer2_weights = num_neurons*num_neurons
        n_output_weights = num_neurons*num_actions

        features_against = np.array(features)
        #print('features are => ',features_against) #
        layer1_weights = regressor[0,0:n_layer1_weights]
        layer1_weights = layer1_weights.reshape(num_inputs, num_neurons)
        #print(layer1_weights.shape)
        layer2_weights = regressor[0,n_layer1_weights:n_layer1_weights+n_layer2_weights]
        layer2_weights = layer2_weights.reshape(num_neurons, num_neurons)
        output_weights = regressor[0,n_layer1_weights+n_layer2_weights:n_layer1_weights+n_layer2_weights+n_output_weights]
        output_weights = output_weights.reshape(num_neurons, num_actions)
        #print(output_weights)
        #print(np.matmul(features_against, layer1_weights))
        layer1_output = f_relu(np.matmul(features_against, layer1_weights))
        #print(layer1_output.shape) #
        layer2_output = f_relu(np.matmul(layer1_output, layer2_weights))
        #print(layer2_output.shape) #
        mult=np.matmul(layer2_output, output_weights)
        #print("CHECK ME",mult)
        prob_orig = f_softmax(mult)
        #print('probabilty => ',prob_orig)
        #print('sum of all prob. => ',np.sum(prob_orig))
        predicted_volt_level=f_decodelabels(prob_orig)
        #print('pred volt level =>',predicted_volt_level)
 
        return predicted_volt_level


#########################################
def f_decodelabels(data):
	max_index = np.argmax(data)
	level = voltage_levels[max_index]
	return level



##########################################


def initialize_func:
	start = time.time()
	voltage_levels=[0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
	#8 levels
	fol='/net/ugrads/gnarang/pvt/dvfi/bo/'


	###############
	## initial x ##
	x_init=np.zeros((4,5)) # 4 VFI anf 5 features
	for i in range(4):
		#x_init[i]=[0.32,0.67, 0.45, 0.90, 0.65] #you can use random too
        	x_init[i] = np.random.rand(1,5)
	print('seed/initial x given => ',x_init)

	x=np.zeros((4,5))

	##############


	num_inputs=5
	num_neurons=20
	num_actions=8
	
	n_layer1_weights = num_inputs*num_neurons
	n_layer2_weights = num_neurons*num_neurons
	n_output_weights = num_neurons*num_actions
	
	regressor_size=n_layer1_weights+n_layer2_weights+n_output_weights
	#weights
	regressor=np.random.rand(1,regressor_size)
	#print("regressor is",regressor)
	
	#end of initialize function
	#return
	

####################


###################


def volt_func(f,w1,w2,w3,w4):
	volt_func_out=np.zeros((4,1))

	for i in range(4):# 4 VFI, 4 policy
		if i==0:		
			regressor=w1
		elif i==1:
			regressor=w2	
		elif i==2:
			regressor=w3	
		else:
			regressor=w4
		####			
		volt_func_out[i] = f_pred_direct(f[i], regressor , num_inputs, num_neurons, num_actions)
	return volt_func_out
###################




def function_evaluate_policy(policy_vfi1, policy_vfi2, policy_vfi3, policy_vfi4 , workload):

	workload=workload
	#workload=['fft', 'canneal', 'dedup', 'fluid', 'lu', 'radix', 'vips', 'water']
	#workload=['fft']
	
	w1=policy_vfi1
	w2=policy_vfi2
	w3=policy_vfi3
	w4=policy_vfi4

	initialize_func()

	for k in range(len(workload)):
		print('======='+workload[k]+'=======')
		print('\n\n')
	
		fl=fol+workload[k]+'/'
		#print(fl)
		if os.path.isdir(fl):
			#remove it and create new
			print('exists, rm and mkdir')
			os.system('rm -rf '+fl)
			os.system('mkdir -p '+fl)
		else: #make it
			print('doesnt exist, just mkdir')
			os.system('mkdir -p '+fl)
	
		folder_out = fl 				# dump runs data in this
		data_folder = fol+'wonje_data/'+workload[k]+'/' 	#pick gem5 data from this
	
	
	
		#depending on workload, a will be different , data_folder is dependent on workload
		a=np.loadtxt(data_folder+'power10.txt')
		num_epoch,cores=a.shape
		#print(a.shape)
	
	
	
	
	
		for epoch in range(num_epoch): # 0 to N-1
			###############
			#voltage prediction
		
			#volt is list of 4 voltages corresponding to VFI
	
			#predict voltage for each VFI (1 to 4) as function of features of each VFI
		
			if epoch==0:
				print('======epoch========',epoch)
				volt=volt_func(x_init, w1,w2,w3,w4) # random seed voltage
				########		
				#rows, cols = (5, 1)
				#volt = [[0.7]*cols]*rows # fixed seed voltage #doesnt work for flatten fuction below
				#print('voltage is', volt)
			else:
				print('=======epoch=======',epoch)
				volt=volt_func(x, w1,w2,w3,w4)
				#print('voltage is', volt)
			###############
			#print('--print volt--')
			#print('=>voltage VFI-1 is', volt[3][0])
		
			print(volt.flatten())
			#np.savetxt(folder_out+'voltages_epoch'+str(int(epoch)+1)+'.txt', volt.flatten())
		
			'''
			# go to gem5 working dir 
			os.chdir('/net/ugrads/gnarang/pvt/gem5/gem5n')
			print("=>changing to gem5 directory")
	
		
			#update fs.py based on voltage predicted
			print('=>updating input script to gem5')
			os.system('./fs_file_gen.py '+str(volt[0][0])+' '+ str(volt[1][0])+' '+ str(volt[2][0])+' '+ str(volt[3][0])+' ')
	
			# takes updated fs_vfi_new, next epoch checkpoint
			gem5_cmd= './build/X86/gem5.fast --redirect-stdout --stdout-file=stdout'+str(epoch+1)+'.txt --redirect-stderr --stderr-file=stderr'+str(epoch+1)+'.txt --outdir='+folder_out+' configs/example/fs_vfi_new.py --disk-image=/net/ugrads/gnarang/pvt/gem5/gem5n/disks/disk_09162021_1035/linux-x86.img --kernel=/net/ugrads/gnarang/pvt/gem5/gem5n/binaries/vmlinux_32_no_floppy --script=/net/ugrads/gnarang/pvt/gem5/gem5n/configs/boot/fft_gn.rcS --cpu-clock=2.5GHz --cpu-type=DerivO3CPU --num-cpus=64 --num-l2cache=64 --num-dirs=64 --l1i_size=32kB --l1d_size=32kB --l2_size=256kB --restore-with-cpu=TimingSimpleCPU --checkpoint-dir='+chkpt_dir+' --checkpoint-restore='+str(epoch+1)+' --ruby --topology=Mesh_XY --network=garnet2.0 --mesh-rows=8 --ruby-clock=2.5GHz -I 20000'
	
		
			print("=>run gem5")
			#os.system('gcc --version')
			#os.system('ldd --version')
			os.system(gem5_cmd)
	
			print("=>gem5 run complete for voltage V")
			'''
			##############################################
			
			# extract and process Fij, IPC for next epoch prediction
			print("=>grepping next features: Fij, IPC\n")
			os.system('./feature_extraction.py '+str(volt[0][0])+' '+ str(volt[1][0])+' '+ str(volt[2][0])+' '+ str(volt[3][0])+' '+str(epoch)+' '+folder_out+' '+data_folder+'') # add workload option later
	
		
			print('=> load extracted freatures\n')
			ftr=np.loadtxt(folder_out+'features'+str(epoch+1)+'.txt')
			#print(ftr.shape)
		
			#append previous voltage volt to ftr
			#append volt list to column of ftr, so ftr will be 4x5 2d array (which is x for next epoch run)
	
			#print(volt)
			volt_transform=volt.flatten()
			#print(volt_transform)
			#print('x shape is ',x.shape)
			x[:,0:4] = ftr  # fij, ipc to first 4 columns [:,i:j+1] so goes 0 to 3
			x[:,4] = volt_transform # previous voltage to last column i.e. 5th column
			print('=>next feature set\n',x)
		
	
			##################
			# edp and execution time calculation
			##################
	
			pwr_array= np.zeros((1,num_epoch))
			pwr_array[0,epoch] = np.loadtxt(folder_out+'power'+str(epoch+1)+'.txt')
			#print("power array",pwr_array)
		
			simsec_array= np.zeros((1,num_epoch))
			simsec_array[0,epoch] = np.loadtxt(folder_out+'simsec'+str(epoch+1)+'.txt')
		
			energy_array= np.zeros((1,num_epoch))
			energy_array[0,epoch] = pwr_array[0,epoch] * simsec_array[0,epoch]
	
			#give dummy x
			#for i in range(4):
	        	#	x[i] = np.random.rand(1,5)
			#print('Next x => ',x)
	
			# Go to DVFI dir
			#os.chdir('/net/ugrads/gnarang/pvt/dvfi')
			#print("back to dvfi folder")
		
			print('\n\n')
	
	
		
		energy_sum = np.sum(energy_array)
		print("=> total Energy for N epochs=",energy_sum)
		np.savetxt(folder_out+'energy.txt', [energy_sum])
	
		simsec_sum = np.sum(simsec_array)
		np.savetxt(folder_out+'execution_time.txt', [simsec_sum])
	
		total_edp = energy_sum * simsec_sum
		np.savetxt(folder_out+'total_edp.txt', [total_edp])
	
		print("=> Execution time=", simsec_sum)
		print("=> Total EDP=", total_edp)
	
	#######################
	# end of workload loop
	#######################
	end = time.time()
	print('=> Evaluate Program time elapsed=' , end - start)

	return total_edp, simsec_sum # add PPW ?




