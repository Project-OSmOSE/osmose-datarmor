# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 11:09:34 2022

@author: gabri
"""
import numpy as np
import matplotlib.pyplot as plt

'''
Include Functions compute and plot Evaluation Metrics to evaluate detection performance 
From : John A. Hildebrand, Kaitlin E. Frasier, Tyler A. Helble, and Marie A. Roch , "Performance metrics for marine mammal signal detection and classification", The Journal of the Acoustical Society of America 151, 414-427 (2022) https://doi.org/10.1121/10.0009270

Librairies : Please, check the file "requierments.txt"

functions here : 
        - ComputeEvaluationMetrics() : computation from detection results
        - plot_PR_curve() : Plot and save Precision-Recall curve
        - plot_DET_curve() : Plot and save Detection-error-tradeoff curve
        - plot_ROC_curve() : Plot and save Receiver-operating-characteristic curve
        - plot_COST_curve() : Plot and save Cost curves
        
        
Paths needed : 
        - path_osmose_dataset : path of the datset with OSmOSE architecture
        - codes_path : path with all the functions needed 
'''

def ComputeEvaluationMetrics(LabelsList, labels, outputs):
    
    '''
        INPUTS :
            - LabelsList : List of label detected by the detector
            - labels : Reference label for each file - Ground Truth (0 or 1)
            - outputs : Outputs of the detector for each file (between 0 and 1)        
    '''
    
    threshold_array = np.linspace(-0.01,1.1,200)
    Recall = np.zeros([np.size(threshold_array), len(LabelsList)])
    Precision = np.zeros_like(Recall)
    TP_rate = np.zeros_like(Recall)
    FP_rate = np.zeros_like(Recall)
    FN_rate = np.zeros_like(Recall)
    
    Npp = 100
    NormalizedExpectedCost = np.zeros([np.size(threshold_array),len(LabelsList), Npp])
        
    for ite in range(np.size(threshold_array)):
        TrueDetection = np.zeros(len(LabelsList))
        MissDetection = np.zeros(len(LabelsList))
        FalseDetection = np.zeros(len(LabelsList))
        TrueAbsence = np.zeros(len(LabelsList))
        
        npTruth = labels.copy()
        npPredict = outputs.copy()
            
        npPredict[npPredict >= threshold_array[ite]] = 1
        npPredict[npPredict <  threshold_array[ite]] = 0
            
        for spectro_id in range(np.size(labels,0)):
            for id_specie in range(len(LabelsList)):
                if npPredict[spectro_id,id_specie] == 1:
                    if npTruth[spectro_id,id_specie] == 1:
                        TrueDetection[id_specie] += 1 #TP
                    if npTruth[spectro_id,id_specie] == 0:
                        FalseDetection[id_specie] += 1  #FP
                if npPredict[spectro_id,id_specie] == 0:
                    if npTruth[spectro_id,id_specie] == 1:
                        MissDetection[id_specie] += 1  #FN
                    if npTruth[spectro_id,id_specie] == 0:
                        TrueAbsence[id_specie] += 1  #TN
                    
        for id_specie in range(len(LabelsList)):            
            if (TrueDetection[id_specie]+FalseDetection[id_specie]) == 0:
                pres = 1
            else :
                pres = TrueDetection[id_specie]/(TrueDetection[id_specie]+FalseDetection[id_specie])
            rec = TrueDetection[id_specie]/(MissDetection[id_specie]+TrueDetection[id_specie])
                
            fpr = FalseDetection[id_specie] / (FalseDetection[id_specie]+TrueAbsence[id_specie])
            tpr = TrueDetection[id_specie] / (TrueDetection[id_specie] + MissDetection[id_specie])
            fnr = MissDetection[id_specie] / (MissDetection[id_specie]+TrueDetection[id_specie])
            
            Recall[ite, id_specie] = rec
            Precision[ite, id_specie] = pres
            FP_rate[ite, id_specie] = fpr
            TP_rate[ite, id_specie] = tpr
            FN_rate[ite, id_specie] = fnr
            
            
            
            Pp = np.linspace(0,1,Npp)
            
            NormalizedExpectedCost[ite, id_specie,:] = Pp * (FN_rate[ite, id_specie]-FP_rate[ite, id_specie]) + FP_rate[ite, id_specie]
    return Recall, Precision, FP_rate, TP_rate, FN_rate, NormalizedExpectedCost, Pp


def plot_PR_curve(Recall, Precision, savepath = None, color='b', xlim=[0,1], ylim=[0,1], figsize=(4,4)):
    plt.figure(figsize=figsize)
    plt.plot(Recall, Precision, c=color)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #plt.title('Best : P = ' + str(Precision[argbest]) + ' - R = ' + str(Recall[argbest]))
    plt.grid()
    plt.tight_layout()
    if not savepath == 'None':
        plt.savefig(savepath)
        
def plot_ROC_curve(FP_rate, TP_rate, savepath = None, color='b', xlim=[0,1], ylim=[0,1], figsize=(4,4)):  
    plt.figure(figsize=figsize)
    plt.plot(FP_rate, TP_rate, c=color)
    plt.xlim(xlim)
    plt.ylim(ylim)
    plt.xlabel('FP rate')
    plt.ylabel('TP rate')
    #plt.title('Best : P = ' + str(Precision[argbest]) + ' - R = ' + str(Recall[argbest]))
    plt.grid()
    plt.tight_layout()
    if not savepath == 'None':
        plt.savefig(savepath)

def plot_DET_curve(FP_rate, FN_rate, savepath = None, color='b', xlim=[0.005,0.8], ylim=[0.005,0.8], figsize=(4,4)):     
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(FP_rate, FN_rate, c=color)
    plt.xlabel('FP rate')
    plt.ylabel('FN rate')
    ax.set_xscale('log')
    ax.set_yscale('log')
    #plt.title('Best : P = ' + str(Precision[argbest]) + ' - R = ' + str(Recall[argbest]))
    plt.grid(True, which="both", ls="-", color='0.65')
    plt.tight_layout()
    if not savepath == 'None':
        plt.savefig(savepath)

def plot_COST_curve(ProbabilityCost, NormalizedExpectedCost, savepath = None, color='b', xlim=[0,1], ylim=[0,0.5], figsize=(4,4)):       
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1,1,1)
    ax.plot(ProbabilityCost, NormalizedExpectedCost, lw=0.1, color=color)
    ax.plot([0,1],[0,1], 'k')
    ax.plot([0,1],[1,0], 'k')
    ax.plot(ProbabilityCost, np.min(NormalizedExpectedCost, axis=1), lw=4, color=color)
    ax.grid()
    ax.set_xlabel('Probability Cost')
    ax.set_ylabel('Normalized Expected Cost')  
    ax.set_ylim([0,0.5]) 
    ax.set_xlim([0,1]) 
    plt.tight_layout()
    if not savepath == 'None':
        plt.savefig(savepath)
        
        
