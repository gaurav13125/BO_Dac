#!/net/ugrads/gnarang/pvt/mypyth/bin/python

# Canneal M3D data
# want to plot avg fij/ipc variation of each VFI over the epochs
# pick the data from workload/voltages.txt and plot

import numpy as np
import matplotlib.pyplot as plt
import pandas
#volt=np.loadtxt('/net/ugrads/gnarang/pvt/dvfi/validat/canneal/voltages.txt')

ipc_vfi1=[]
fij_vfi1=[]

ipc_vfi2=[]
fij_vfi2=[]

ipc_vfi3=[]
fij_vfi3=[]

ipc_vfi4=[]
fij_vfi4=[]

epoch=[]

for e in range(955): # 955 ?
	f1=open('/net/ugrads/gnarang/pvt/dvfi/validat/canneal/features'+str(e+1)+'.txt')
	line = f1.readlines()
	line = [x.strip() for x in line]
	#line_len=len(line) 	
	#print(line)
	vfi1=line[0]
	vfi2=line[1]
	vfi3=line[2]
	vfi4=line[3]
	#print(vfi1)
	#print(vfi2)
	#print(vfi3)
	#print(vfi4)
	
	sp=vfi1.split()
	ipc_vfi1.append(sp[1]) # avg ipc
	fij_vfi1.append(sp[3]) # avg fij


	sp=vfi2.split()
	ipc_vfi2.append(sp[1]) # avg ipc
	fij_vfi2.append(sp[3]) # avg fij

	sp=vfi3.split()
	ipc_vfi3.append(sp[1]) # avg ipc
	fij_vfi3.append(sp[3]) # avg fij

	sp=vfi4.split()
	ipc_vfi4.append(sp[1]) # avg ipc
	fij_vfi4.append(sp[3]) # avg fij
	
	epoch.append(e)
	f1.close()


df = pandas.DataFrame(data={"col1": epoch, "col2": ipc_vfi1, "col3": ipc_vfi2, "col4": ipc_vfi3, "col5": ipc_vfi4, "col6": fij_vfi1, "col7": fij_vfi2, "col8": fij_vfi3, "col9": fij_vfi4})
df.to_csv("./canneal_m3d_trained_w.csv", sep=',',index=False)

#print(len(ipc_vfi4))
#print(fij_vfi4)
'''
vfi1=[]
vfi2=[]
vfi3=[]
vfi4=[]
epoch=[]

for i in range(line_len):
	v=line[i]
	v1=v.rstrip(']')
	v2=v1.lstrip('[')
	v3=v2.split() # list of 4 voltages (strings)

	vfi1.append(float(v3[0]))
	vfi2.append(float(v3[1]))
	vfi3.append(float(v3[2]))
	vfi4.append(float(v3[3]))
	epoch.append(i)
f1.close()

#print(vfi1)

###############

fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(epoch, fij_vfi1)
axs[0, 0].set_title('VFI1')
axs[0, 1].plot(epoch, fij_vfi2, 'tab:orange')
axs[0, 1].set_title('VFI2')
axs[1, 0].plot(epoch, fij_vfi3, 'tab:green')
axs[1, 0].set_title('VFI3')
axs[1, 1].plot(epoch, fij_vfi4, 'tab:red')
axs[1, 1].set_title('VFI4')

for ax in axs.flat:
    ax.set(xlabel='epoch', ylabel='fij')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()

plt.show()
'''
####################################
'''
plt.plot(epoch,fij_vfi1)
#plt.plot(epoch,vfi1,'k.-', epoch,vfi2,'b.-', epoch,vfi3,'r.-', epoch,vfi4,'m.-')
#plt.legend(('voltage-vfi-1', 'voltage-vfi-2', 'voltage-vfi-3', 'voltage-vfi-4'),
#           loc='lower center', shadow=True)
plt.ylabel('Fij')
plt.xlabel('Epoch')
plt.title('m3d - canneal')

#plt.axis([.0001, 10000])

#plt.ylim(0,1)
plt.show()
'''

#############

