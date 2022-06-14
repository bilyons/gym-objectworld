import random
import numpy as np
import os
import sys
from scipy.special import softmax
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt

vairldfs = []
vairlpoldfs = []
iqldfs = []
iavidfs = []
for i in range(5):
	vairldfs.append(np.genfromtxt(os.path.abspath(os.getcwd())+"/data/{}/vairl.csv".format(i), delimiter=','))
	vairlpoldfs.append(np.genfromtxt(os.path.abspath(os.getcwd())+"/data/{}/vairl_pol.csv".format(i), delimiter=','))
	iqldfs.append(np.genfromtxt(os.path.abspath(os.getcwd())+"/data/{}/iql.csv".format(i), delimiter=','))
	iavidfs.append(np.genfromtxt(os.path.abspath(os.getcwd())+"/data/{}/iavi.csv".format(i), delimiter=','))

VAIRLEVD = np.zeros((len(vairldfs[0]), 5))
VAIRLPR = np.zeros((len(vairldfs[0]), 5))
VAIRLTruePR = np.zeros((len(vairldfs[0]), 5))
VAIRLRuntime = np.zeros((len(vairldfs[0]), 5))

VAIRLPOLEVD = np.zeros((len(vairlpoldfs[0]), 5))
VAIRLPOLPR = np.zeros((len(vairlpoldfs[0]), 5))
VAIRLPOLTruePR = np.zeros((len(vairlpoldfs[0]), 5))
VAIRLPOLRuntime = np.zeros((len(vairlpoldfs[0]), 5))

IQLEVD = np.zeros((len(iqldfs[0]), 5))
IQLPR = np.zeros((len(iqldfs[0]), 5))
IQLTruePR = np.zeros((len(iqldfs[0]), 5))
IQLRuntime = np.zeros((len(iqldfs[0]), 5))


IAVIEVD = np.zeros((len(iavidfs[0]), 5))
IAVIPR = np.zeros((len(iavidfs[0]), 5))
IAVITruePR = np.zeros((len(iqldfs[0]), 5))
IAVIRuntime = np.zeros((len(iavidfs[0]), 5))

for j in range(1, len(vairldfs[0]-1)):
	for i in range(len(vairldfs)):
		VAIRLEVD[j, i] = vairldfs[i][j,1]
		VAIRLPR[j, i] = vairldfs[i][j,2]
		VAIRLTruePR[j, i] = vairldfs[i][j,3]
		VAIRLRuntime[j, i] = vairldfs[i][j,5]

		VAIRLPOLEVD[j, i] = vairlpoldfs[i][j,1]
		VAIRLPOLPR[j, i] = vairlpoldfs[i][j,2]
		VAIRLPOLTruePR[j, i] = vairlpoldfs[i][j,3]
		VAIRLPOLRuntime[j, i] = vairlpoldfs[i][j,5]

		IQLEVD[j, i] = iqldfs[i][j,1]
		IQLPR[j, i] = iqldfs[i][j,2]
		IQLTruePR[j, i] = iqldfs[i][j, 3]
		IQLRuntime[j, i] = iqldfs[i][j,5]

		IAVIEVD[j, i] = iavidfs[i][j,1]
		IAVIPR[j, i] = iavidfs[i][j,2]
		IAVITruePR[j, i] = iavidfs[i][j, 3]
		IAVIRuntime[j, i] = iavidfs[i][j,5]

VAIRLEVDMean = np.mean(VAIRLEVD, axis=1)
VAIRLEVDStd = np.std(VAIRLEVD, axis=1)

VAIRLPRMean = np.mean(VAIRLTruePR, axis=1)
VAIRLPRStd = np.std(VAIRLTruePR, axis=1)

VAIRLTruePRMean = np.mean(VAIRLTruePR, axis=1)
VAIRLTruePRStd = np.std(VAIRLTruePR, axis=1)

VAIRLRuntimeMean = np.mean(VAIRLRuntime, axis=1)
VAIRLRuntimeStd = np.std(VAIRLRuntime, axis=1)

VAIRLPOLEVDMean = np.mean(VAIRLPOLEVD, axis=1)
VAIRLPOLEVDStd = np.std(VAIRLPOLEVD, axis=1)

VAIRLPOLPRMean = np.mean(VAIRLPOLPR, axis=1)
VAIRLPOLPRStd = np.std(VAIRLPOLPR, axis=1)

VAIRLPOLTruePRMean = np.mean(VAIRLPOLTruePR, axis=1)
VAIRLPOLTruePRStd = np.std(VAIRLPOLTruePR, axis=1)

VAIRLPOLRuntimeMean = np.mean(VAIRLPOLRuntime, axis=1)
VAIRLPOLRuntimeStd = np.std(VAIRLPOLRuntime, axis=1)

IQLEVDMean = np.mean(IQLEVD, axis=1)
IQLEVDStd = np.std(IQLEVD, axis=1)

IQLPRMean = np.mean(IQLPR, axis=1)
IQLPRStd = np.std(IQLPR, axis=1)

IQLTruePRMean = np.mean(IQLTruePR, axis=1)
IQLTruePRStd = np.std(IQLTruePR, axis=1)

IQLRuntimeMean = np.mean(IQLRuntime, axis=1)
IQLRuntimeStd = np.std(IQLRuntime, axis=1)

IAVIEVDMean = np.mean(IAVIEVD, axis=1)
IAVIEVDStd = np.std(IAVIEVD, axis=1)

IAVIPRMean = np.mean(IAVIPR, axis=1)
IAVIPRStd = np.std(IAVIPR, axis=1)

IAVITruePRMean = np.mean(IAVITruePR, axis=1)
IAVITruePRStd = np.std(IAVITruePR, axis=1)

IAVIRuntimeMean = np.mean(IAVIRuntime, axis=1)
IAVIRuntimeStd = np.std(IAVIRuntime, axis=1)

x = [1,2,4,8,16,32,64,128,256,512,1024]

f, ax = plt.subplots(1)
# ax.plot(x_m, y_m, 'or')
ax.plot(x, VAIRLEVDMean[1:], '-', label='VAIRL')
ax.plot(x, VAIRLPOLEVDMean[1:], '-', label='VAIRLPOL')
ax.plot(x, IQLEVDMean[1:], '-', label='IQL')
ax.plot(x, IAVIEVDMean[1:], '-', label='IAVI')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.fill_between(x, VAIRLEVDMean[1:] - VAIRLEVDStd[1:], VAIRLEVDMean[1:] + VAIRLEVDStd[1:],alpha=0.1)
ax.fill_between(x, VAIRLPOLEVDMean[1:] - VAIRLPOLEVDStd[1:], VAIRLPOLEVDMean[1:] + VAIRLPOLEVDStd[1:],alpha=0.1)
ax.fill_between(x, IQLEVDMean[1:] - IQLEVDStd[1:], IQLEVDMean[1:] + IQLEVDStd[1:],alpha=0.1)
ax.fill_between(x, IAVIEVDMean[1:] - IAVIEVDStd[1:], IAVIEVDMean[1:] + IAVIEVDStd[1:],alpha=0.1)
plt.xlabel("Number of trajectories")
plt.ylabel("EVD")
plt.show()

f, ax = plt.subplots(1)
# ax.plot(x_m, y_m, 'or')
ax.plot(x, VAIRLPRMean[1:], '-', label='VAIRL')
ax.plot(x, VAIRLPOLPRMean[1:], '-', label='VAIRLPOL')
ax.plot(x, IQLPRMean[1:], '-', label='IQL')
ax.plot(x, IAVIPRMean[1:], '-', label='IAVI')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.fill_between(x, VAIRLPRMean[1:] - VAIRLPRStd[1:], VAIRLPRMean[1:] + VAIRLPRStd[1:],alpha=0.1)
ax.fill_between(x, VAIRLPOLPRMean[1:] - VAIRLPOLPRStd[1:], VAIRLPOLPRMean[1:] + VAIRLPOLPRStd[1:],alpha=0.1)
ax.fill_between(x, IQLPRMean[1:] - IQLPRStd[1:], IQLPRMean[1:] + IQLPRStd[1:],alpha=0.1)
ax.fill_between(x, IAVIPRMean[1:] - IAVIPRStd[1:], IAVIPRMean[1:] + IAVIPRStd[1:],alpha=0.1)
plt.xlabel("Number of trajectories")
plt.ylabel("Pearson Rank")
plt.show()

f, ax = plt.subplots(1)
# ax.plot(x_m, y_m, 'or')
ax.plot(x, VAIRLPRMean[1:], '-', label='VAIRL')
ax.plot(x, VAIRLPOLTruePRMean[1:], '-', label='VAIRLPOL')
ax.plot(x, IQLTruePRMean[1:], '-', label='IQL')
ax.plot(x, IAVITruePRMean[1:], '-', label='IAVI')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.fill_between(x, VAIRLPRMean[1:] - VAIRLPRStd[1:], VAIRLPRMean[1:] + VAIRLPRStd[1:],alpha=0.1)
ax.fill_between(x, VAIRLPOLTruePRMean[1:] - VAIRLPOLTruePRStd[1:], VAIRLPOLTruePRMean[1:] + VAIRLPOLTruePRStd[1:],alpha=0.1)
ax.fill_between(x, IQLTruePRMean[1:] - IQLTruePRStd[1:], IQLTruePRMean[1:] + IQLTruePRStd[1:],alpha=0.1)
ax.fill_between(x, IAVITruePRMean[1:] - IAVITruePRStd[1:], IAVITruePRMean[1:] + IAVITruePRStd[1:],alpha=0.1)
plt.xlabel("Number of trajectories")
plt.ylabel("True Pearson Rank")
plt.show()

f, ax = plt.subplots(1)
# ax.plot(x_m, y_m, 'or')
ax.plot(x, VAIRLRuntimeMean[1:], '-', label='VAIRL')
ax.plot(x, VAIRLPOLRuntimeMean[1:], '-', label='VAIRLPOL')
ax.plot(x, IQLRuntimeMean[1:], '-', label='IQL')
ax.plot(x, IAVIRuntimeMean[1:], '-', label='IAVI')
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles[::-1], labels[::-1])
ax.fill_between(x, VAIRLRuntimeMean[1:] - VAIRLRuntimeStd[1:], VAIRLRuntimeMean[1:] + VAIRLRuntimeStd[1:],alpha=0.1)
ax.fill_between(x, VAIRLPOLRuntimeMean[1:] - VAIRLPOLRuntimeStd[1:], VAIRLPOLRuntimeMean[1:] + VAIRLPOLRuntimeStd[1:],alpha=0.1)
ax.fill_between(x, IQLRuntimeMean[1:] - IQLRuntimeStd[1:], IQLRuntimeMean[1:] + IQLRuntimeStd[1:],alpha=0.1)
ax.fill_between(x, IAVIRuntimeMean[1:] - IAVIRuntimeStd[1:], IAVIRuntimeMean[1:] + IAVIRuntimeStd[1:],alpha=0.1)
plt.yscale("log")
plt.xlabel("Number of trajectories")
plt.ylabel("Runtime (s)")
plt.show()