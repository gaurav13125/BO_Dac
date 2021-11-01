#!/net/ugrads/gnarang/pvt/mypyth/bin/python

# want to plot voltage variation of each VFI over the epochs
# pick the data from workload/voltages.txt and plot

import numpy as np
import matplotlib.pyplot as plt
import pandas

#volt=np.loadtxt('/net/ugrads/gnarang/pvt/dvfi/validat/canneal/voltages.txt')

f1=open('/net/ugrads/gnarang/pvt/dvfi/validat/canneal/voltages.txt')
line = f1.readlines()
line = [x.strip() for x in line]
line_len=len(line) 	
#print(line)

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


df = pandas.DataFrame(data={"epoch": epoch, "volt_vfi1": vfi1, "volt_vfi2": vfi2, "volt_vfi3": vfi3, "volt_vfi4": vfi4})
df.to_csv("./canneal_m3d_trained_w_voltages.csv", sep=',',index=False)

#print(vfi1)
'''
#plt.plot(epoch,vfi1)
plt.plot(epoch,vfi1,'k.-', epoch,vfi2,'b.-', epoch,vfi3,'r.-', epoch,vfi4,'m.-')
plt.legend(('voltage-vfi-1', 'voltage-vfi-2', 'voltage-vfi-3', 'voltage-vfi-4'),
           loc='lower center', shadow=True)
plt.ylabel('Voltage level')
plt.xlabel('Epoch')
plt.title('voltage vs epoch - canneal')

#plt.axis([.0001, 10000])

plt.ylim(0,1)
plt.show()
'''
###############
'''
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(epoch, vfi1)
axs[0, 0].set_title('VFI1')
axs[0, 1].plot(epoch, vfi2, 'tab:orange')
axs[0, 1].set_title('VFI2')
axs[1, 0].plot(epoch, vfi3, 'tab:green')
axs[1, 0].set_title('VFI3')
axs[1, 1].plot(epoch, vfi4, 'tab:red')
axs[1, 1].set_title('VFI4')

for ax in axs.flat:
    ax.set(xlabel='epoch', ylabel='voltage')

# Hide x labels and tick labels for top plots and y ticks for right plots.
#for ax in axs.flat:
#    ax.label_outer()

plt.show()
'''

#########

