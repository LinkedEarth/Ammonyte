# -*- coding: utf-8 -*-
"""
N. Ahmad, S. Derrible, T. Eason, and H. Cabezas, 2016, “Using Fisher information to track stability in multivariate systems”,
Royal Society Open Science, 3:160582, DOI: 10.1098/rsos.160582

Edits have been made by Alexander James to streamline the code for usage in this package
"""

import pandas as pd 
import math
import numpy as np
import pyleoclim as pyleo

__all__ = [
    'fisher_information',
    'smooth_series'
]

def fisher_information(eig_data,w_size,w_incre):
    '''Compute Fisher Information from Laplacian eigenmap data using sliding windows

    Implements the Fisher Information metric following Ahmad et al. (2016), adapted
    for use with Laplacian eigenmaps derived from recurrence matrix analysis.

    Parameters
    ----------

    eig_data : list
        List of rows where each row is [time, eig1, eig2, eig3, eig4, ...].
        The first element of each row is a time value and the remaining elements
        are eigenmap coordinates.

    w_size : int
        Window size (number of data points per window).

    w_incre : int
        Window increment (step size between successive windows).

    Returns
    -------

    time_axis : numpy.ndarray
        Time values corresponding to each Fisher Information estimate.

    values : numpy.ndarray
        Fisher Information values for each window.

    References
    ----------

    .. [1] N. Ahmad, S. Derrible, T. Eason, and H. Cabezas (2016).
           Using Fisher information to track stability in multivariate systems.
           Royal Society Open Science, 3:160582.
           DOI: 10.1098/rsos.160582
    '''
    Data_num=[]
    Time=[]
    
    for row in eig_data:
        Time.append(row[0])
        temp=[]
        for i in range(1,len(row)):
            if row[i]=='':
                temp.append(0)
            else:
                temp.append(float(row[i]))
        Data_num.append(temp)
        
    sost_data = SOST(eig_data,w_size)
    sost = sost_data.values[0]
    
    FI_final=[]
    k_init=[]
    for i in range(0,len(Data_num),w_incre):
        
        Data_win=Data_num[i:i+w_size]
        win_number=i
        
        if len(Data_win)==w_size:
            Bin=[]
            for m in range(len( Data_win)):
                Bin_temp=[]
                
                for n in range(len( Data_win)):
                    if m==n:
                        Bin_temp.append('I')
                    else:
                        Bin_temp_1=[]
                    
                        for k in range(len(Data_win[n])):
                            if (abs(Data_win[m][k]-Data_win[n][k]))<=sost[k]:
                                Bin_temp_1.append(1)
                            else:
                                Bin_temp_1.append(0)
                                
                        Bin_temp.append(sum(Bin_temp_1))
                        
                Bin.append(Bin_temp)
            
            FI=[]
            for tl in range(1,101):
                tl1=len(sost)*float(tl)/100
                Bin_1=[]
                Bin_2=[]
                
                for j in range(len(Bin)):
                    if j not in Bin_2:
                       
                        Bin_1_temp=[j]
                        for i in range(len(Bin[j])):
                            if Bin[j][i]!='I' and Bin[j][i]>=tl1 and i not in Bin_2:
                                Bin_1_temp.append(i)
                                
                        Bin_1.append(Bin_1_temp)
                        Bin_2.extend(Bin_1_temp)

                prob=[0]
                for i in Bin_1:
                    prob.append(float(len(i))/len(Bin_2))
                    
                prob.append(0)
                
                prob_q=[]
                for i in prob:
                    prob_q.append(math.sqrt(i))
                    
                FI_temp=0
                for i in range(len(prob_q)-1):
                    FI_temp+=(prob_q[i]-prob_q[i+1])**2
                FI_temp=4*FI_temp    
                
                FI.append(FI_temp)
                
            for i in range(len(FI)):
                if FI[i]!=8.0:
                    k_init.append(FI.index(FI[i]))
                    break
                
            FI_final.append(FI)
            
    if len(k_init)==0:
        k_init.append(0)
        
    time_axis = []
    
    for i in range(0,len(FI_final)):
        FI_final[i].append(float(sum(FI_final[i][min(k_init):len(FI_final[i])]))/len(FI_final[i][min(k_init):len(FI_final[i])]))
        time_axis.append(Time[(i*w_incre+w_size)-1])
        
    FI_final = pd.DataFrame(FI_final)
    values = FI_final.iloc[:,-1]
    
    return np.array(time_axis), np.array(values)
        
def SOST(eig_data,s_for_sd):
    '''Compute the Sum of Standard Deviations (SOST) threshold for Fisher Information binning

    Calculates the minimum sliding-window standard deviation for each eigenmap axis,
    doubled, to serve as the bin-width threshold used in the Fisher Information computation.

    Parameters
    ----------

    eig_data : list
        List of rows in the same format as fisher_information: each row is
        [time, eig1, eig2, ...].

    s_for_sd : int
        Window size used when computing the sliding standard deviation.

    Returns
    -------

    df_sos : pandas.DataFrame
        Single-row DataFrame where each column contains the SOST threshold
        for the corresponding eigenmap axis.

    See Also
    --------

    ammonyte.utils.fisher.fisher_information
    '''
    Data_num=[]
    
    for row in eig_data:
        temp=[]
        for i in range(1,len(row)):
            if row[i]=='':
                temp.append(0)
            else:
                temp.append(float(row[i]))
        Data_num.append(temp)
        
    df=pd.DataFrame(Data_num)
    
    sos=[]
    for j in range(len(df.columns)):
        sos_temp=[]
        for i in df.index:
            A=list(df[j][i:i+s_for_sd])
            A_1=[float(i) for i in A if i!=0 ]
        
            if len(A_1)==s_for_sd:
                sos_temp.append(np.std(A_1,ddof=1))
                
        if len(sos_temp)==0:
            sos.append(0)
        else:
            
            sos.append(min(sos_temp)*2)
        
    df_sos=pd.DataFrame(sos)
    df_sos=df_sos.transpose()
    
    return df_sos
    
def smooth_series(series,block_size):
    '''Function to smooth series by averaging chunks
    
    Parameters
    ----------
    
    series : pyleoclim.Series object
        Series to be smoothed
        
    block_size : int
        Size of each block. Assumes an evenly spaced series.

    Returns
    -------

    series : ammonyte.RQARes or pyleoclim.Series
        The input series with values replaced by block-averaged values.
        Note: modifies and returns the input series in-place.
    '''
    
    values = series.value
    
    smoothed_values=[]
   
    for i in range(block_size,len(values)+block_size,block_size):
        for _ in range(i-block_size,i):
            smoothed_values.append(float(sum(values[i-block_size:i]))/len(values[i-block_size:i]))
            
    smoothed_values=smoothed_values[0:len(values)]
    
    series.value = np.array(smoothed_values)
    
    return series