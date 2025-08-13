# Inspired from KJStrand's Pulse_Rate_Estimation https://github.com/KJStrand/Pulse_Rate_Estimation
# Code structure:
#   Imports
#   Algorithm parameters
#   Function definitions:
#       Loading datasets
#       Evaluate(): Evaluating an algorithm on a dataset (returns error metric and plots graphs)
#       RunPulseRateAlgorithm(): Evaluating an algorithm on a file from a dataset
#       AnalyzeWindow(): Predict heart rate with an algorithm on a PPG signal window. 
#        This is where the different algorithms are defined.
#       Utils (scaling, filtering, applying FFT, ...)
#   Launching the evaluation of the algorithms
#   Analyzing a single window, visualizing the datasets

# ## Part 1: Pulse Rate Algorithm
# 
# ### Dataset
# **Troika**[1] dataset is used to develop the motion-compensated pulse rate algorithm.
# **Capnobase** is used to confirm that an algorithm works on clean data.
# Troika has accelerometer data to help for motion compensation, but CapnoBase does not. CapnoBase has no motion artefacts anyway.
# 
# 1. Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. Link
# 
# 2. W. Karlen, S. Raman, J. M. Ansermino and G. A. Dumont, "Multiparameter Respiratory Rate Estimation From the Photoplethysmogram," in IEEE Transactions on Biomedical Engineering, vol. 60, no. 7, pp. 1946-1953, July 2013, doi: 10.1109/TBME.2013.2246160.
# -----


# ### Code


import glob

import numpy as np
import scipy as sp

import matplotlib.pyplot as plt
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 25


plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=MEDIUM_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from scipy.signal import find_peaks
from scipy.interpolate import interp1d
from scipy.stats import zscore
import pandas as pd
import random
import ctypes

import heartpy as hp
import biobss as bb # Doesn't work on Windows ? :'( -> On WSL/Linux it works

### Parameters begin ###

# Don't bother reading these parameters if you haven't read the rest of the code first

import warnings
warnings.simplefilter("ignore") # Disables warnings caused by HeartPy

# Choose the dataset
dataset = "capnobase"
# dataset = "troika"
plotResults = True

# Algorithm parameters
scaling_on = True # True to enable a simple signal scaling on the ppg and acc signals, substracting the mean then dividing by (max-min)
filter_on = True # True to enable a scipy butterworth order 3 bandpass filter on ppg and acc signals
minBeats = 2 # For algorithms based on signal peak detection.

### Peak detection parameters for scipy find_peaks on ppg time-based signal:
## Without dependencies on sampling frequency (deprecated). Uncomment one of the first three and the fourth line
# findPeaksArgsTime = {"height":10, "distance":35} # Parameters of the original github code by KJStrand (troika dataset)
# findPeaksArgsTime = {"prominence":9} # Fiddle for better perf on troika (not conclusive...)
# findPeaksArgsTime = {"prominence":3} # Parameters that seem good for capnobase dataset
# get_find_peaks_args_time = lambda fs: findPeaksArgsTime 
## With dependency on sampling frequency. Uncomment one of the three
## Distance such that 185 BPM max
# get_find_peaks_args_time = lambda fs: {"height":0.7,"distance":int(60/200*fs)} # Works best on capnobase
# get_find_peaks_args_time = lambda fs: {"height":0.00,"distance":int(60/200*fs)} # Works best on troika
get_find_peaks_args_time = lambda fs: {"height":0.3,"distance":int(60/200*fs)} # Tradeoff troika-capnobase

# Peak detection parameters for scipy find_peaks on frequency signals (see FFT_Peak method)
findPeaksArgsFreq = {"height":185} # Parameters of the original github code by KJStrand (troika dataset)

# Biobss parameters
# Optimised for CapnoBase
# biobssPeakDelta = 0.8 # delta parameter for BIOBSS' peakdet method, when using ppg_detectpeaks
# biobssBeatDelta = 0.07 # delta parameter for BIOBSS' peakdet method, when using ppg_detectpeaks
# Optimised for Troika
# biobssPeakDelta = 0.5 # delta parameter for BIOBSS' peakdet method, when using ppg_detectpeaks
# biobssBeatDelta = 0.13 # delta parameter for BIOBSS' peakdet method, when using ppg_detectpeaks
# Tradeoff
biobssPeakDelta = 0.65 # delta parameter for BIOBSS' peakdet method, when using ppg_detectpeaks
biobssBeatDelta = 0.09 # delta parameter for BIOBSS' peakdet method, when using ppg_detectpeaks

# Tsai's "2thresh" threshold
# From https://github.com/Mic-Tsai/Sensor_PPG_MAX30102
# threshTsai = 0.6 # Capno
# threshTsai = 0.7 # Troika
threshTsai = 0.65 # Tradeoff

zThresh = 2 # Outlier removal for HR computations based on peaks

### Parameters end ###


# Interfacing with C language, in particular the example algorithm given by Maxim when buying the MAX30101WING
# https://pgi-jcns.fz-juelich.de/portal/pages/using-c-from-python.html
# _algo_HR = ctypes.CDLL("./libalgoHR_wfilter.so")
_algo_HR = ctypes.CDLL("./libalgo_HR.so")
_algo_HR.HRSpO2Func.argtypes = (ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ushort,
                ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_short), 
                ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_ushort),ctypes.POINTER(ctypes.c_ushort),ctypes.POINTER(ctypes.c_ushort))

def HRSpO2Func(dinIR,dinRed,dinGreen,ns,SampRate,compSpO2,
                ir_ac_comp,red_ac_comp,green_ac_comp,ir_ac_mag,red_ac_mag, 
                green_ac_mag,HRbpm2,SpO2B,DRdy):
    global _algo_HR
    _algo_HR.HRSpO2Func(dinIR,dinRed,dinGreen,ns,SampRate,compSpO2,
                ctypes.byref(ir_ac_comp),ctypes.byref(red_ac_comp), ctypes.byref(green_ac_comp), ctypes.byref(ir_ac_mag),ctypes.byref(red_ac_mag), 
                ctypes.byref(green_ac_mag),ctypes.byref(HRbpm2),ctypes.byref(SpO2B),ctypes.byref(DRdy))


def LoadCapnobaseDataset():
    """Retrieve the filenames of the Capnobase dataset, located at ./datasets/capnobase/data"""
    data_dir = "./datasets/capnobase/data"
    # Data without any artefacts
    data_fls = sorted(glob.glob(data_dir + "/*_signal.csv"))
    ref_fls = sorted(glob.glob(data_dir + "/*_reference.csv"))
    # Data with a few artefacts
    # data_fls = sorted(glob.glob(data_dir+"/data_with_artifs/*_signal.csv"))
    # ref_fls = sorted(glob.glob(data_dir+"/data_with_artifs/*_reference.csv"))
    # Both
    # data_fls = sorted(glob.glob(data_dir + "/*_signal.csv")+glob.glob(data_dir+"/data_with_artifs/*_signal.csv"))
    # ref_fls = sorted(glob.glob(data_dir + "/*_reference.csv")+glob.glob(data_dir+"/data_with_artifs/*_reference.csv"))
    return data_fls, ref_fls

def LoadCapnobaseDataFile(data_fl,ref_fl):
    """
    Load the signal and the reference HR from a Capnobase file (an entry from LoadCapnobaseDataset output list)
    
    Returns: two numpy arrays, the first is the signal, the second is the reference
    """
    data = pd.read_csv(data_fl)["pleth_y"].to_numpy()
    temp = pd.read_csv(ref_fl)
    hrX = np.array(temp["hr_ecg_x"].to_numpy()[0].split(" ")[1:],dtype=np.float64)
    hrY = np.array(temp["hr_ecg_y"].to_numpy()[0].split(" ")[1:],dtype=np.float64)
    ref = np.array([hrX,hrY])
    return data,ref

def LoadTroikaDataset():
    """
    Retrieve the .mat filenames for the troika dataset.

    Review the README in ./datasets/troika/ to understand the organization of the .mat files.

    Returns:
        data_fls: Names of the .mat files that contain signal data
        ref_fls: Names of the .mat files that contain reference data
        <data_fls> and <ref_fls> are ordered correspondingly, so that ref_fls[5] is the 
            reference data for data_fls[5], etc...
    """
    data_dir = "./datasets/troika/training_data"
    data_fls = sorted(glob.glob(data_dir + "/DATA_*.mat"))
    ref_fls = sorted(glob.glob(data_dir + "/REF_*.mat"))
    return data_fls, ref_fls

def LoadTroikaDataFile(data_fl):
    """
    Loads and extracts signals from a troika data file.

    Usage:
        data_fls, ref_fls = LoadTroikaDataset()
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fls[0])

    Args:
        data_fl: (str) filepath to a troika .mat file.

    Returns:
        numpy arrays for ppg, accx, accy, accz signals.
    """
    data = sp.io.loadmat(data_fl)['sig']
    return data[2:]

def AggregateErrorMetric(pr_errors, gnd_truths, confidence_est):
    """
    Computes an aggregate error metric based on confidence estimates.

    Args:
        pr_errors: a numpy array of errors between pulse rate estimates and corresponding 
            reference heart rates.
        gnd_truths: a numpy array of ground truth HR in BPM for each pulse rate error
        confidence_est: a numpy array of confidence estimates for each pulse rate
            error.

    Returns:
        Some error metrics, whatever you want (just uncomment)
    """
    # Higher confidence means a better estimate. The best 90% of the estimates
    #    are above the 10th percentile confidence.
    # percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    # best_estimates = pr_errors[confidence_est >= percentile90_confidence]
    # best_truths = gnd_truths[confidence_est >= percentile90_confidence]

    # return np.mean(np.abs(best_estimates)) # Mean absolute error
    # return np.sqrt(np.mean(best_estimates**2)) # RMS
    # return np.mean(100*np.abs(best_estimates)/best_truths) # in percents
    # return (np.mean(np.abs(best_estimates)),np.mean(np.abs(best_estimates)/best_truths)) # both
    return (np.mean(np.abs(pr_errors)),np.mean(np.abs(pr_errors)/gnd_truths)) # both but without the 90 percentile stuff
    # return (np.median(np.abs(best_estimates)),np.median(np.abs(best_estimates)/best_truths)) # both medians
    # return (np.std(best_estimates),np.std(best_estimates/best_truths)) # both stds


def Evaluate(method):
    """
    Top-level algorithm evaluation function.

    Runs the pulse rate algorithm on the dataset and returns an aggregate error metric.

    Returns:
        Pulse rate error metric on the dataset. See AggregateErrorMetric.
    """
    # Retrieve dataset files
    global dataset
    if dataset == "capnobase" :
        data_fls, ref_fls = LoadCapnobaseDataset()
    elif dataset == "troika" :
        data_fls, ref_fls = LoadTroikaDataset()
    else :
        print("Unknown dataset")
        exit(1)

    errs, trths, confs = [], [], []
    for data_fl, ref_fl in list(zip(data_fls, ref_fls))[:]: # For each file
        # Run the pulse rate algorithm on each trial in the dataset
        errors, truths, confidence = RunPulseRateAlgorithm(data_fl, ref_fl, method)
        errs.append(errors)
        trths.append(truths)
        confs.append(confidence)

    # Compute aggregate error metric 
    errs = np.hstack(errs)
    trths = np.hstack(trths)
    confs = np.hstack(confs)
    preds = errs + trths
    errsPercent = errs * 100 / trths
    # percentile90_confidence = np.percentile(confs, 10)
    # errMean = np.mean(errs[confs >= percentile90_confidence])
    # errSTD = np.std(errs[confs >= percentile90_confidence])
    errMean = np.mean(errs)
    errSTD = np.std(errs)
    
    # plt.figure(figsize=(7,5.2)) # Histogram Ground Truth HR
    # plt.hist(trths,20)
    # plt.grid()
    # plt.xlabel("Ground truth BPM [BPM]")
    # plt.ylabel("Occurences")
    # plt.title(f"Ground Truth HR in TROIKA")
    # plt.savefig(f"./images/GTHR_troika.pdf")
    # plt.close()

    # Plots
    if plotResults :
        plt.figure(figsize=(8,5.5)) # Error BPM vs ground truth BPM
        plt.plot(trths, errs,linestyle="None",marker=".",color=(0,0,0.9,0.4))
        # plt.plot(trths[confs >= percentile90_confidence], errs[confs >= percentile90_confidence],linestyle="None",marker=".",color=(0,0,0.9,0.5))
        # plt.plot(trths[confs < percentile90_confidence], errs[confs < percentile90_confidence], linestyle="None", marker="x",color=(0.9,0,0,1))
        plt.hlines([errMean],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0.9,0,0,0.7),linestyles="dashdot",label="$\mu$")
        plt.hlines([errMean-1.96*errSTD,errMean+1.96*errSTD],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0.9,0,0,0.7),linestyles="dotted",label="$\mu \pm 1.96 \sigma$")
        # plt.hlines([errMean-1.96*errSTD,errMean+1.96*errSTD],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0,0.7,0.75,1),linestyles="dotted")
        # plt.hlines([errMean],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0,0.7,0.75,1),linestyles="dashdot")
        plt.plot([0,np.max(trths)+10],[0,-np.max(trths)-10],label="$HR_{pred} = 0$",color=(0,0.7,0,0.7))
        plt.xlim(left=np.min(trths)-10,right=np.max(trths)+10)
        plt.ylim(bottom=np.min(errs)-15,top=np.max(errs)+15)
        plt.grid()
        plt.xlabel("Ground truth HR [BPM]")
        plt.ylabel("Error ($HR_{pred}-HR_{true}$) [BPM]")
        plt.legend()
        plt.title(f"{method}: Error vs. Ground Truth HR")
        plt.savefig(f"./images/{dataset}/Err_{method}.pdf")
        plt.close()

        plt.figure(figsize=(7,5.5)) # Predicted BPM vs ground truth BPM
        plt.plot(trths, preds,linestyle="None",marker=".",color=(0,0,0.9,0.5))
        # plt.plot(trths[confs >= percentile90_confidence], errs[confs >= percentile90_confidence],linestyle="None",marker=".",color=(0,0,0.9,0.5))
        # plt.plot(trths[confs < percentile90_confidence], errs[confs < percentile90_confidence], linestyle="None", marker="x",color=(0.9,0,0,1))
        # plt.hlines([errMean],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0.9,0,0,0.7),linestyles="dashdot",label="$\mu$")
        # plt.hlines([errMean-1.96*errSTD,errMean+1.96*errSTD],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0.9,0,0,0.7),linestyles="dotted",label="$\mu \pm 1.96 \sigma$")
        # plt.hlines([errMean-1.96*errSTD,errMean+1.96*errSTD],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0,0.7,0.75,1),linestyles="dotted")
        # plt.hlines([errMean],xmin=np.min(trths)-10,xmax=np.max(trths)+10,colors=(0,0.7,0.75,1),linestyles="dashdot")
        plt.plot([0,np.max(preds)+10],[0,np.max(preds)+10],label="Error = 0",color=(0,0.7,0,0.4))
        plt.xlim(left=np.min(trths)-10,right=np.max(trths)+10)
        plt.ylim(bottom=np.min(preds)-15,top=np.max(preds)+15)
        plt.grid()
        plt.xlabel("Ground truth HR [BPM]")
        plt.ylabel("Prediction [BPM]")
        plt.legend()
        plt.title(f"{method}: Prediction vs. Ground Truth HR")
        plt.savefig(f"./images/{dataset}/Pred_{method}.pdf")
        plt.close()
        
        plt.figure(figsize=(7,5.5)) # Error % vs ground truth BPM
        # plt.plot(trths[confs >= percentile90_confidence], errsPercent[confs >= percentile90_confidence],linestyle="None",marker=".",color=(0,0,0.9,0.5))
        # plt.plot(trths[confs < percentile90_confidence], errsPercent[confs < percentile90_confidence], linestyle="None", marker="x",color=(0.9,0,0,1))
        plt.plot(trths, errsPercent,linestyle="None",marker=".",color=(0,0,0.9,0.5))
        plt.grid()
        plt.xlabel("Ground truth HR [BPM]")
        plt.ylabel("Error [%]")
        plt.title(f"{method}")
        plt.savefig(f"./images/{dataset}/PErr_{method}.pdf")
        plt.close()

        plt.figure(figsize=(7,5.5)) # Histogram error
        plt.hist(errs,20)
        plt.grid()
        plt.xlabel("Error [BPM]")
        plt.ylabel("Occurences")
        plt.title(f"{method}")
        plt.savefig(f"./images/{dataset}/Hist_{method}.pdf")
        plt.close()

        plt.figure(figsize=(7,5.5)) # Zoomed histogram error
        plt.hist(errs[np.abs(errs) < 30],20)
        plt.grid()
        plt.xlabel("Error [BPM]")
        plt.ylabel("Occurences")
        plt.title(f"{method}")
        plt.savefig(f"./images/{dataset}/ZHist_{method}.pdf")
        plt.close()

        plt.figure(figsize=(7,5.5)) # Histogram error %
        plt.hist(errsPercent,20)
        plt.grid()
        plt.xlabel("Error [%]")
        plt.ylabel("Occurences")
        plt.title(f"{method}")
        plt.savefig(f"./images/{dataset}/PHist_{method}.pdf")
        plt.close()

        plt.figure(figsize=(7,5.5)) # Zommed histogram error %
        plt.hist(errsPercent[np.abs(errsPercent) < 30],20)
        plt.grid()
        plt.xlabel("Error [%]")
        plt.ylabel("Occurences")
        plt.title(f"{method}")
        plt.savefig(f"./images/{dataset}/PZHist_{method}.pdf")
        plt.close()

    return AggregateErrorMetric(errs, trths, confs)

def RunPulseRateAlgorithm(data_fl, ref_fl, method):
    '''Given a sample data file with PPG (and 3 accelerometer in the case of troika) channels and reference file with ground truth heart rates,
       compute pulse rates every two seconds.
       Parameters:
           data_fl: filename of file containing PPG and X, Y, Z accelerometer data from dataset
           ref_fl: filename of file containing ground truth heart rates from dataset
       
       Returns:
           errors: numpy array with differences between predicted and reference heart rates
           confidence: numpy array with confidence values for heart rate predictions
    '''
    
    global dataset
    if dataset == "troika" :
        Fs = 125 # Troika data has sampling rate of 125 Hz, Capnobase 300 Hz
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
        ref = sp.io.loadmat(ref_fl)
        
    elif dataset == "capnobase" :
        Fs = 300
        ppg, ref = LoadCapnobaseDataFile(data_fl, ref_fl)
        # Resample to another frequency
        # targetFs = 125
        # ppg = interp1d(np.arange(len(ppg))/Fs, ppg)(np.arange(len(ppg)*targetFs/Fs)/targetFs)
        # Fs = targetFs
        accx = np.zeros_like(ppg)
        accy = accx
        accz = accx
    else :
        print("Unknown dataset")
        exit(1)
    
    winSize = 8*Fs # Ground truth BPM provided in 8 second windows
    winShift = 2*Fs # Successive ground truth windows overlap by 2 seconds
    errs = []
    confs = []
    trths = []
    
    # For each 8 second window, compute a predicted BPM and confidence and compare to ground truth
    offset = 0
    # print(f"Number of windows: {len(ref['BPM0'])}")
    # minBPM = 300
    # maxBPM = 0
    if dataset == "troika":
        nWindows = len(ref['BPM0'])
    elif dataset == "capnobase":
        if len(ppg) < winSize : # Just a check but shouldn't happen
            return [],[],[]
        nWindows = (len(ppg)-winSize)//winShift + 1
    else :
        print("Unknown dataset")
        exit(1)

    for eval_window_idx in range(nWindows):
        
        # Set verbose to True to visualize plot analysis
        verbose = False
        # verbose = True if eval_window_idx == 28 else False
    
        window_start = offset
        window_end = winSize+offset
        offset += winShift
        
        if verbose:
            print(f"Win start,end: {window_start}, {window_end}")
        
        ppg_window = ppg[window_start:window_end]
        accx_window = accx[window_start:window_end]
        accy_window = accy[window_start:window_end]
        accz_window = accz[window_start:window_end]

        try :
            pred, conf = AnalyzeWindow(ppg_window, accx_window, accy_window, accz_window, method, Fs=Fs, verbose=verbose)
            # if conf < 0.0001 : 
            #   print(data_fl)
            #   print(eval_window_idx)
        except RuntimeWarning: # Heartpy being annoying, requires the warnings turned into errors (see imports at the top)
            print("Runtime warning")
            print(data_fl)
            print(eval_window_idx)
            pred = 0
            conf = -3e6
        except hp.exceptions.BadSignalWarning: # Heartpy being really annoying
            print("Bad signal warning")
            print(data_fl)
            print(eval_window_idx)
            pred = 0
            conf = -3e6
        except KeyboardInterrupt:
            exit(1)
        
        if dataset == "capnobase":
            refWindow = ref[1,(ref[0,:] >= window_start/Fs) & (ref[0,:] < window_end/Fs)]
            if len(refWindow) == 0 : continue
            groundTruthBPM = 1/np.mean(1/refWindow) # 60*Fs/mean(diff(peaks)) = 60*Fs/mean(60*Fs/HRref) = 1/mean(1/HRref)
        elif dataset == "troika":
            groundTruthBPM = ref['BPM0'][eval_window_idx][0]
        else :
            print("Unknown dataset")
            exit(1)

        if verbose:
            print('Ground Truth BPM: ', groundTruthBPM)
        # if groundTruthBPM < minBPM : minBPM = groundTruthBPM
        # if groundTruthBPM > maxBPM : maxBPM = groundTruthBPM

        predError = pred - groundTruthBPM
        errs.append(predError)
        trths.append(groundTruthBPM)
        confs.append(conf)

    # print("min, max: ", minBPM, " ", maxBPM)
    errors, truths, confidence = np.array(errs), np.array(trths), np.array(confs)
    return errors, truths, confidence

def AnalyzeWindow(ppg, accx, accy, accz, method, Fs=125, verbose=False):
    ''' Analyze a single 8 second window of PPG and Accelerometer data.
        Parameters:
            ppg: numpy array with ppg values
            accx/y/z: numpy arrays with per-axis accelerometer data
            Fs: sampling rate used by both PPG and accelerometer sensors
            verbose: display plots and logging information.
    
        Returns:
            prediction: Tuple of (BPM prediction, confidence) for this window.
    '''
    
    global get_find_peaks_args_time
    global findPeaksArgsFreq
    global filter_on
    global scaling_on

    # Preprocessing
    ppg_preproc = ppg
    if filter_on:
        ppg_preproc = BandpassFilter(ppg_preproc, fs=Fs)
    if scaling_on:
        ppg_preproc = ScaleSignal(ppg_preproc)

    # Aggregate accelerometer data into single signal
    
    accx_preproc = BandpassFilter(accx, fs=Fs)
    accy_preproc = BandpassFilter(accy, fs=Fs)
    accz_preproc = BandpassFilter(accz, fs=Fs)
    accy_mean = accy-np.mean(accy_preproc) # Center Y values
    acc_mag = np.sqrt(accx_preproc**2+accy_mean**2+accz_preproc**2)
    # acc_mag = ScaleSignal(acc_mag)
    acc_mag = BandpassFilter(acc_mag, fs=Fs)
    
    prediction = 0
    confidence = 0

    if method == "FFT_Peak":

        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0]
            
        # Use FFT length larger than the input signal size for higher spectral resolution.
        fft_len=len(ppg_preproc)*4

        # Create an array of frequency bins
        freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031

        # The frequencies between 40 BPM and 240 BPM Hz
        low_freqs = (freqs >= (40/60)) & (freqs <= (240/60))
        
        mag_freq_ppg, fft_ppg = FreqTransform(ppg_preproc, freqs, low_freqs, fft_len)
        mag_freq_acc, _ = FreqTransform(acc_mag, freqs, low_freqs, fft_len)
        
        peaks_ppg = find_peaks(mag_freq_ppg, **findPeaksArgsFreq)[0]
        peaks_acc = find_peaks(mag_freq_acc, **findPeaksArgsFreq)[0]
        
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True)
        sorted_freq_peaks_acc = sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True)
        
        # Use the frequency peak with the highest magnitude.
        if len(sorted_freq_peaks_ppg) > 0 :
            use_peak = sorted_freq_peaks_ppg[0]

            chosen_freq = freqs[low_freqs][use_peak]
            prediction = chosen_freq * 60
            confidence = CalcConfidence(chosen_freq, freqs, fft_ppg)
        else :
            print("No freq")
            use_peak = 0
            chosen_freq = 0

    elif method == "FFT_Peak_Acc":

        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0]
    
        if verbose:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
            ax1.title.set_text('Signal with Time Domain FindPeaks()')
            ax1.plot(ppg_preproc)
            ax1.plot(peaks, ppg_preproc[peaks], "x")
            
            ax2.title.set_text('Aggregated Accelerometer Data')
            ax2.plot(acc_mag, color="purple")
            plt.savefig("./images/signal.pdf")
            plt.close()
            # plt.show()
            
        # Use FFT length larger than the input signal size for higher spectral resolution.
        fft_len=len(ppg_preproc)*4

        # Create an array of frequency bins
        freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031
        df = freqs[1]

        # The frequencies between 40 BPM and 240 BPM Hz
        low_freqs = (freqs >= (40/60)) & (freqs <= (240/60))
        offset = freqs[low_freqs][0]

        mag_freq_ppg, fft_ppg = FreqTransform(ppg_preproc, freqs, low_freqs, fft_len)
        mag_freq_acc, _ = FreqTransform(acc_mag, freqs, low_freqs, fft_len)
        
        peaks_ppg = find_peaks(mag_freq_ppg, **findPeaksArgsFreq)[0]
        peaks_acc = find_peaks(mag_freq_acc, **findPeaksArgsFreq)[0]
        # peaks_acc = find_peaks(mag_freq_acc, height=40)[0]
        
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = np.array(sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True))
        sorted_freq_peaks_acc = np.array(sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True))
        
        # Use the frequency peak with the highest magnitude, unless the peak is also present in the accelerometer peaks.
        if len(sorted_freq_peaks_ppg) > 0 :
            use_peak = sorted_freq_peaks_ppg[0]
            use_i = 0
            for i in range(len(sorted_freq_peaks_ppg)):
                # Check nearest two peaks also
                cond1 = sorted_freq_peaks_ppg[i] in sorted_freq_peaks_acc
                cond2 = sorted_freq_peaks_ppg[i]-1 in sorted_freq_peaks_acc
                cond3 = sorted_freq_peaks_ppg[i]+1 in sorted_freq_peaks_acc
                if cond1 or cond2 or cond3:
                    continue
                else:
                    use_peak = sorted_freq_peaks_ppg[i]
                    use_i = i
                    break

            chosen_freq = freqs[low_freqs][use_peak]
            prediction = chosen_freq * 60
            confidence = CalcConfidence(chosen_freq, freqs, fft_ppg)
        else :
            print("No freq")
            use_peak = 0
            chosen_freq = 0
        
        if verbose:
            plt.figure(figsize=(6.4,5))
            plt.title("PPG Frequency Spectrum")
            plt.plot(freqs[low_freqs], mag_freq_ppg,label="Spectrum magnitude")
            plt.plot(peaks_ppg*df+offset, mag_freq_ppg[peaks_ppg], marker="x", linestyle="None", color="orange", markersize=8, label="Unused peaks")
            plt.vlines(sorted_freq_peaks_ppg[:use_i]*df+offset,ymin=0,ymax=mag_freq_ppg[sorted_freq_peaks_ppg[:use_i]],colors="red",linestyles="dashed")
            plt.plot(sorted_freq_peaks_ppg[:use_i]*df+offset, mag_freq_ppg[sorted_freq_peaks_ppg[:use_i]], "xr", markersize=8,label="Motion artefacts")
            plt.vlines(use_peak*df+offset,ymin=0,ymax=mag_freq_ppg[use_peak], colors="green", linestyles="dashed")
            plt.plot(use_peak*df+offset, mag_freq_ppg[use_peak], "xg", markersize=8, label="Used peak frequency")
            plt.xlim(left=offset,right=freqs[low_freqs][-1])
            plt.ylim(bottom=0)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Spectrum magnitude")
            plt.grid()
            plt.legend()
            plt.savefig("./images/PPG Freq.pdf")
            plt.close()
            # plt.show()
            
            plt.figure(figsize=(6.4,5))
            plt.title("Acceleration Frequency Spectrum")
            plt.plot(freqs[low_freqs], mag_freq_acc,color="purple",label="Spectrum magnitude")
            plt.plot(peaks_acc*df+offset, mag_freq_acc[peaks_acc], marker="x", linestyle="None", color="orange", markersize=8)
            plt.vlines(sorted_freq_peaks_ppg[:use_i]*df+offset,ymin=0,ymax=mag_freq_acc[sorted_freq_peaks_ppg[:use_i]],colors="red",linestyles="dashed")
            plt.plot(sorted_freq_peaks_ppg[:use_i]*df+offset, mag_freq_acc[sorted_freq_peaks_ppg[:use_i]], "xr", markersize=8)
            plt.vlines(use_peak*df+offset,ymin=0,ymax=mag_freq_acc[use_peak], colors="green", linestyles="dashed")
            plt.plot(use_peak*df+offset, mag_freq_acc[use_peak], "xg", markersize=8)
            plt.xlim(left=offset,right=freqs[low_freqs][-1])
            plt.ylim(bottom=0)
            plt.xlabel("Frequency [Hz]")
            plt.ylabel("Spectrum magnitude")
            plt.grid()
            plt.savefig("./images/ACC Freq.pdf")
            plt.close()
            # plt.show()
            
            print("PPG Freq Peaks: ", peaks_ppg)
            print("ACC Freq Peaks: ", peaks_acc)
            
            print("PPG Freq Peaks Sorted: ", sorted_freq_peaks_ppg)
            print("ACC Freq Peaks Sorted: ", sorted_freq_peaks_acc)
            print("Use peak: ", use_peak)
            print(f"Predicted BPM: {prediction}, {chosen_freq} (Hz), Confidence: {confidence}")      

    elif method == "BIOBSS Peaks":
        peaks = bb.ppgtools.ppg_detectpeaks(ppg_preproc, Fs, delta=biobssPeakDelta)["Peak_locs"] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.mean(np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1 
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Peaks PostMean":
        peaks = bb.ppgtools.ppg_detectpeaks(ppg_preproc, Fs, delta=biobssPeakDelta)["Peak_locs"] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = np.mean(Fs/np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1 
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Peaks Median":
        peaks = bb.ppgtools.ppg_detectpeaks(ppg_preproc, Fs, delta=biobssPeakDelta)["Peak_locs"] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.median(np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Peaks Outlier":
        peaks = bb.ppgtools.ppg_detectpeaks(ppg_preproc, Fs, delta=biobssPeakDelta)["Peak_locs"] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            intervals = np.diff(peaks)
            prediction = Fs/np.mean(intervals[np.abs(zscore(intervals)) < zThresh]) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Beats":
        peaks = bb.ppgtools.ppg_detectbeats(ppg_preproc, Fs, delta=biobssBeatDelta) # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.mean(np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1 
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Beats PostMean":
        peaks = bb.ppgtools.ppg_detectbeats(ppg_preproc, Fs, delta=biobssBeatDelta) # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = np.mean(Fs/np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1 
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Beats Median":
        peaks = bb.ppgtools.ppg_detectbeats(ppg_preproc, Fs, delta=biobssBeatDelta) # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.median(np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "BIOBSS Beats Outlier":
        peaks = bb.ppgtools.ppg_detectbeats(ppg_preproc, Fs, delta=biobssBeatDelta) # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            intervals = np.diff(peaks)
            prediction = Fs/np.mean(intervals[np.abs(zscore(intervals)) < zThresh]) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "Peaktime_diff":
        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.mean(np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1 # more intervals means more confidence -> actually punishes low BPMs for no reason
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "Peaktime_diff_PostMean":
        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = np.mean(Fs/np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "Peaktime_diff_Median":
        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.median(np.diff(peaks),overwrite_input=True) * 60
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "Peaktime_diff_Outlier":
        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            intervals = np.diff(peaks)
            prediction = Fs/np.mean(intervals[np.abs(zscore(intervals)) < zThresh]) * 60
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "2thresh":
        lowerThreshold = -threshTsai
        upperThreshold = threshTsai
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = 0
        nBeats = 0
        for i in range(len(ppg_preproc)) :
            if ppg_preproc[i] < lowerThreshold :
                if beatComplete :
                    prediction += Fs / (i-lastTime) * 60
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_preproc[i] > upperThreshold and beatStarted :
                beatComplete = True
        if nBeats > minBeats : 
            prediction /= nBeats # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        else : 
            # print("No beats")
            prediction = 0
            confidence = 0.0

    elif method == "2thresh_PreMean":
        lowerThreshold = -threshTsai
        upperThreshold = threshTsai
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = 0
        nBeats = 0
        for i in range(len(ppg_preproc)) :
            if ppg_preproc[i] < lowerThreshold :
                if beatComplete :
                    prediction += i-lastTime
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_preproc[i] > upperThreshold and beatStarted :
                beatComplete = True
        if nBeats > minBeats : 
            prediction = Fs / (prediction/nBeats) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        else : 
            # print("No beats")
            prediction = 0
            confidence = 0.0

    elif method == "2thresh_Median":
        lowerThreshold = -threshTsai
        upperThreshold = threshTsai
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = []
        nBeats = 0
        for i in range(len(ppg_preproc)) :
            if ppg_preproc[i] < lowerThreshold :
                if beatComplete :
                    prediction.append(i-lastTime)
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_preproc[i] > upperThreshold and beatStarted :
                beatComplete = True
        if nBeats > minBeats : 
            prediction = Fs / np.median(prediction,overwrite_input=True) * 60
            confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        else : 
            # print("No beats")
            prediction = 0
            confidence = 0.0

    elif method == "2thresh_Outlier":
        lowerThreshold = -threshTsai
        upperThreshold = threshTsai
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = []
        nBeats = 0
        for i in range(len(ppg_preproc)) :
            if ppg_preproc[i] < lowerThreshold :
                if beatComplete :
                    prediction.append(i-lastTime)
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_preproc[i] > upperThreshold and beatStarted :
                beatComplete = True
        if nBeats > minBeats : 
            prediction = np.array(prediction)
            prediction = Fs / np.mean(prediction[np.abs(zscore(prediction)) < zThresh]) * 60
            confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        else : 
            # print("No beats")
            prediction = 0
            confidence = 0.0

    elif method == "NumberOfBeats":
        lowerThreshold = -threshTsai
        upperThreshold = threshTsai
        beatStarted = False
        beatComplete = False
        lastTime = 0
        nBeats = 0
        for i in range(len(ppg_preproc)) :
            if ppg_preproc[i] < lowerThreshold :
                if beatComplete :
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_preproc[i] > upperThreshold and beatStarted :
                beatComplete = True
        # if nBeats > minBeats : 
        prediction = nBeats / (len(ppg_preproc)/Fs) * 60
        confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        # else : 
        #     # print("No beats")
        #     prediction = 0
        #     confidence = 0.0

    elif method == "NumberOfPeaks":
        peaks = find_peaks(ppg_preproc, **get_find_peaks_args_time(Fs))[0]
        prediction = len(peaks) / (len(ppg_preproc)/Fs) * 60
        confidence = 1
        # confidence = 1 - 1/(len(peaks)+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        if len(peaks) == 0 : 
            prediction = 0
            confidence = 0
            # print("No peaks")

    elif method == "MAX FTHR":
        
        ppg_bp_max = np.max(ppg_preproc)
        ppg_bp_min = np.min(ppg_preproc)

        if ppg_bp_max - ppg_bp_min < 1e-9 :
            # print("No amplitude")
            prediction = 0
            confidence = 0

        else :

            #(ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ushort,
            #        ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_short), 
            #        ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_ushort),ctypes.POINTER(ctypes.c_ushort),ctypes.POINTER(ctypes.c_ushort))

            dinIR = ctypes.c_uint()
            dinRed = ctypes.c_uint()
            dinGreen = ctypes.c_uint()
            ns = ctypes.c_uint()
            SampRate = ctypes.c_ushort(Fs)
            compSpO2 = ctypes.c_ushort(1)

            ir_ac_comp = ctypes.c_short(0)
            red_ac_comp = ctypes.c_short(0)
            green_ac_comp = ctypes.c_short(0)
            ir_ac_mag = ctypes.c_short(0)
            red_ac_mag = ctypes.c_short(0)

            green_ac_mag = ctypes.c_short(0)
            HRbpm2 = ctypes.c_ushort(0)
            SpO2B = ctypes.c_ushort(0)
            DRdy = ctypes.c_ushort(0)

            nPredictions = 0
            prediction = 0

            scaleFactor = 2**15 / (ppg_bp_max-ppg_bp_min)
            shift = (ppg_bp_max+ppg_bp_min)/2
            # print(shift)
            # print(scaleFactor)

            ppg_MAX = (ppg_preproc - shift)*scaleFactor + 2**18
            # if np.size(ppg_MAX[ppg_MAX < 0]) > 0 : print("Ca pue")
            # if np.size(ppg_MAX[ppg_MAX > 2**32-1]) > 0 : print("Ca pue")

            # Resample to 100 Hz
            ppg_MAX = interp1d(np.arange(len(ppg_MAX))/Fs, ppg_MAX)(np.arange(len(ppg_MAX)*100/Fs)/100)

            for i in range(len(ppg_MAX)):

                dinIR.value = int(ppg_MAX[i])
                dinRed.value = int(ppg_MAX[i])
                dinGreen.value = int(ppg_MAX[i])
                ns.value = i

                HRSpO2Func(dinIR,dinRed,dinGreen,ns,SampRate,compSpO2,
                        ir_ac_comp,red_ac_comp,green_ac_comp,ir_ac_mag,red_ac_mag, 
                        green_ac_mag,HRbpm2,SpO2B,DRdy)
                
                if DRdy.value == 1 :
                    prediction += HRbpm2.value
                    nPredictions += 1

            if nPredictions > 0 :
                prediction /= nPredictions
                confidence = 1 - 1/(nPredictions+1) # more predictions means more confidence -> actually punishes low BPMs for no reason, so do not use this
            else :
                prediction = 0
                confidence = 0
                # print("No predictions")

    elif method == "MAX FTHR Median":
        
        ppg_bp_max = np.max(ppg_preproc)
        ppg_bp_min = np.min(ppg_preproc)

        if ppg_bp_max - ppg_bp_min < 1e-9 :
            # print("No amplitude")
            prediction = 0
            confidence = 0

        else :

            #(ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_uint, ctypes.c_ushort, ctypes.c_ushort,
            #        ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short), ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_short), 
            #        ctypes.POINTER(ctypes.c_short),ctypes.POINTER(ctypes.c_ushort),ctypes.POINTER(ctypes.c_ushort),ctypes.POINTER(ctypes.c_ushort))

            dinIR = ctypes.c_uint()
            dinRed = ctypes.c_uint()
            dinGreen = ctypes.c_uint()
            ns = ctypes.c_uint()
            SampRate = ctypes.c_ushort(Fs)
            compSpO2 = ctypes.c_ushort(1)

            ir_ac_comp = ctypes.c_short(0)
            red_ac_comp = ctypes.c_short(0)
            green_ac_comp = ctypes.c_short(0)
            ir_ac_mag = ctypes.c_short(0)
            red_ac_mag = ctypes.c_short(0)

            green_ac_mag = ctypes.c_short(0)
            HRbpm2 = ctypes.c_ushort(0)
            SpO2B = ctypes.c_ushort(0)
            DRdy = ctypes.c_ushort(0)

            nPredictions = 0
            prediction = []

            scaleFactor = 2**31 / (ppg_bp_max-ppg_bp_min)
            shift = (ppg_bp_max+ppg_bp_min)/2

            for i in range(len(ppg_preproc)):
                
                ppg_MAX = int((ppg_preproc[i] - shift)*scaleFactor + 2**31)

                dinIR.value = ppg_MAX
                dinRed.value = ppg_MAX
                dinGreen.value = ppg_MAX
                ns.value = i

                HRSpO2Func(dinIR,dinRed,dinGreen,ns,SampRate,compSpO2,
                        ir_ac_comp,red_ac_comp,green_ac_comp,ir_ac_mag,red_ac_mag, 
                        green_ac_mag,HRbpm2,SpO2B,DRdy)
                
                if DRdy.value == 1 :
                    # prediction += HRbpm2.value
                    prediction.append(HRbpm2.value)#*Fs/100)
                    nPredictions += 1

            if nPredictions > 0 :
                prediction = np.median(prediction,overwrite_input=True)
                confidence = 1 - 1/(nPredictions+1) # more predictions means more confidence -> actually punishes low BPMs for no reason, so do not use this
                # confidence = 1
            else :
                prediction = 0
                confidence = 0
                # print("No predictions")

    elif method == "HeartPy" :

        # hp_preproc_ppg = ppg_preproc
        hp_preproc_ppg = hp.scale_data(ppg_preproc)
        # hp_preproc_ppg = hp.hampel_correcter(ppg,Fs) # Terrible, do not use...
        # hp_preproc_ppg = hp.filter_signal(hp_preproc_ppg,[0.5, 6],sample_rate=Fs,order=3,filtertype="bandpass")
        # hp_preproc_ppg = hp.enhance_peaks(hp_preproc_ppg,iterations=2) # Also terrible


        _, m = hp.process(hp_preproc_ppg, Fs)
        prediction = m["bpm"]
        confidence = -m["sdsd"]*m["bpm"]
        if np.isnan(m["bpm"]) :
            prediction = 0
            confidence = -3e6
        elif np.ma.isMA(m["sdsd"]):
            confidence = -2e6
        # hp.plotter(wd,m)
        # plt.show()
        

    elif method == "dummy":
        prediction = random.uniform(50,150)
        # prediction = 87.868
        confidence = 1

    else :
        print("Unknown method")
        exit(1)

        
    return (prediction, confidence)

def ScaleSignal(x):
    x -= np.mean(x)
    if np.std(x) > 1e-9 :
        return x/np.std(x)
    # if abs(max(x)-min(x)) > 1e-9 :
    #     return x/(max(x)-min(x))
    return x 

def BandpassFilter(signal, fs):
    '''Bandpass filter the signal between 40 and 240 BPM'''
    
    # Convert to Hz
    lo, hi = 40/60, 240/60
    
    b, a = sp.signal.butter(3, (lo, hi), btype='bandpass', fs=fs)
    return sp.signal.filtfilt(b, a, signal)

def FreqTransform(x, freqs, low_freqs, fft_len):
    '''Compute and return FFT and magnitude of FFT for given low frequencies
        Parameters:
            x: numpy array input signal to transform
            freqs: full list of FFT frequency bins
            low_freqs: low frequency bins between 40 BPM and 240 BPM
            fft_len: length of FFT to compute
            
        Returns:
            mag_freq_x: magnitude of lower frequencies of the FFT transformed signal
            fft_x: FFT of normalized input signal
    '''
    
    # Take an FFT of the normalized signal
    # norm_x = (x )/(max(x)-min(x))
    norm_x = (x )/np.std(x)
    fft_x = np.fft.rfft(norm_x, fft_len)
    
    # Calculate magnitude of the lower frequencies
    mag_freq_x = np.abs(fft_x)[low_freqs]
    
    return mag_freq_x, fft_x

def CalcConfidence(chosen_freq, freqs, fft_ppg):
    '''Calculates a confidence value for a given frequency by computing
       the ratio of energy concentrated near that frequency compared to the full signal.
       Parameters:
           chosen_freq: frequency prediction for heart rate.
           freqs: full list of FFT frequency bins
           fft_ppg: FFT of normalized PPG signal
       
       Returns:
           conf_val: Confidence value for heart rate prediction.
    '''
    win = (40/60.0)
    win_freqs = (freqs >= chosen_freq - win) & (freqs <= chosen_freq + win)
    abs_fft_ppg = np.abs(fft_ppg)
    
    # Sum frequency spectrum near pulse rate estimate and divide by sum of entire spectrum
    conf_val = np.sum(abs_fft_ppg[win_freqs])/np.sum(abs_fft_ppg)
    
    return conf_val

### Evaluate on all files in dataset ###


# method = "FFT_Peak" # Apply FFT on the window and select most present frequency, convert this frequency to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "FFT_Peak_Acc" # (Original TROIKA code) Apply FFT and select most present frequency (+ remove matches with Accelerometer FFT), convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Peaks" # Use BIOBSS Peak detection, take the mean between peaks and convert it into a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Peaks PostMean" # Use BIOBSS Peak detection, convert time between peaks into BPM values and take the mean of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Peaks Median" #  Use BIOBSS Peak detection, convert time between peaks into BPM values and take the take the median of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Peaks Outlier" #  Use BIOBSS Peak detection, convert time between peaks into BPM values and take the take the median of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# print()
# method = "BIOBSS Beats" # Use BIOBSS Peak detection, take the mean between peaks and convert it into a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Beats PostMean" # Use BIOBSS Peak detection, convert time between peaks into BPM values and take the mean of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Beats Median" #  Use BIOBSS Peak detection, convert time between peaks into BPM values and take the take the median of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "BIOBSS Beats Outlier" #  Use BIOBSS Peak detection, convert time between peaks into BPM values and take the take the median of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# print()
# method = "NumberOfPeaks" # Compute the number of peaks in the window and convert it to a BPM using the length of the window
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "Peaktime_diff" # Compute BPM values based on the time difference between two consecutive peaks, take the mean of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "Peaktime_diff_PostMean" # Compute the mean time difference between peak occurences and convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "Peaktime_diff_Median" # Compute the median time difference between peak occurences and convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "Peaktime_diff_Outlier" # Compute the median time difference between peak occurences and convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# print()
# method = "NumberOfBeats"
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "2thresh" # Compute BPM values based on the duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again), take the mean of the BPM values
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "2thresh_PreMean" # Compute the mean duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again) and convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "2thresh_Median" # Compute the median duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again) and convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "2thresh_Outlier" # Compute the median duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again) and convert it to a BPM
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
method = "MAX FTHR" # Base algorithm on MAX32630 FTHR, with FIR filter, finger off detection and peaktime_diff
MAE = Evaluate(method)
print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "MAX FTHR Median" # Base algorithm on MAX32630 FTHR, with FIR filter, finger off detection and peaktime_diff, take the median of the predictions
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "HeartPy" # HeartPy library
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")
# method = "dummy" # Pick a HR value at random between 50 and 150 BPM (or a constant 88 BPM)
# MAE = Evaluate(method)
# print(f"Method : {method} =====\n MAE is: [{MAE[0]:.3f}],[{MAE[1]*100:.3f}]")


exit(0)


#### Dataset Visualisations ###

# Test visualisation of capsobase dataset
# data, ref = LoadCapnobaseDataFile("./datasets/capnobase/data/0009_8min_signal.csv","./datasets/capnobase/data/0009_8min_reference.csv")
# # print(data.shape)
# # print(ref.shape)
# data= data[5000:7400]
# data = ScaleSignal(BandpassFilter(data, fs=300))
# peaks = find_peaks(data, **get_find_peaks_args_time(300))[0]
# print(peaks)
# peaks = bb.ppgtools.ppg_detectpeaks(data, 300, delta=0.01)
# print(peaks["Peak_locs"])


# print(bb.ppgtools.from_cycles(data,peaks["Peak_locs"], peaks["Trough_locs"], 300)["ppg_PR_mean"])
# print(60*300/np.mean(np.diff(peaks["Peak_locs"])))

# ref = ref[:,ref[0] <= 8]
# print(np.mean(ref[1]))
# plt.figure()
# peaks = find_peaks(data,**get_find_peaks_args_time(300))[0]
# plt.plot(data)
# plt.plot(peaks,data[peaks],"xg")
# plt.savefig("./test-data.pdf")
# plt.close()
# plt.figure()
# plt.plot(ref[0],ref[1])
# plt.savefig("./test-ref.pdf")
# plt.close()
# exit(0)

# data = pd.read_csv("./datasets/capnobase/data/data_with_artifs/0016_8min_signal.csv")
# Fs = 300
# artefLoc = 138226 # from labels file
# ppg = data["pleth_y"].to_numpy()[artefLoc-Fs*4:artefLoc+Fs*8+1]
# ecg = data["ecg_y"].to_numpy()[artefLoc-Fs*4:artefLoc+Fs*8+1]
# ppgBP = ScaleSignal(BandpassFilter(ppg, Fs))
# ppg = ScaleSignal(ppg)
# ecg = ScaleSignal(ecg)
# x = np.arange(len(ppg))/Fs
# fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True, figsize=(8,6))
# ax1.plot(x,ecg,label="ECG")
# ax2.plot(x,ppg,label="PPG",color="orange")
# ax3.plot(x,ppgBP,label="Filtered PPG",color="green")
# plt.xlim(left=x[0],right=x[-1])
# # ax1.set_title("ECG")
# # ax2.set_title("PPG")
# # ax3.set_title("Filtered PPG")
# ax1.set_ylim(top=9)
# ax2.set_ylim(top=2.2)
# ax3.set_ylim(top=3.8)
# ax3.set_xlabel("Time [s]")
# ax1.legend(loc="upper right")
# ax2.legend(loc="upper right")
# ax3.legend(loc="upper right")
# ax1.grid()
# ax2.grid()
# ax3.grid()
# fig.suptitle("Motion Artefact on CapnoBase Dataset")
# fig.savefig("./images/artefsCapno.pdf")
# plt.close()

# exit(0)

# data = sp.io.loadmat("./datasets/troika/training_data/DATA_01_TYPE01.mat")["sig"]
# Fs = 125
# artefLoc = 4519 # from labels file
# ppg = data[2,artefLoc-Fs*4:artefLoc+Fs*4+1]
# ecg = data[0,artefLoc-Fs*4:artefLoc+Fs*4+1]
# ppgBP = ScaleSignal(BandpassFilter(ppg, Fs))
# ppg = ScaleSignal(ppg)
# ecg = ScaleSignal(ecg)
# x = np.arange(len(ppg))/Fs
# fig, (ax1, ax2, ax3) = plt.subplots(3,1,sharex=True, figsize=(8,6))
# ax1.plot(x,ecg,label="ECG")
# ax2.plot(x,ppg,label="PPG",color="orange")
# ax3.plot(x,ppgBP,label="Filtered PPG",color="green")
# plt.xlim(left=x[0],right=x[-1])
# # ax1.set_title("ECG")
# # ax2.set_title("PPG")
# # ax3.set_title("Filtered PPG")
# ax1.set_ylim(top=9)
# ax2.set_ylim(top=2.8)
# ax3.set_ylim(top=2.5)
# ax3.set_xlabel("Time [s]")
# ax1.legend(loc="upper right")
# ax2.legend(loc="lower right")
# ax3.legend(loc="lower right")
# ax1.grid()
# ax2.grid()
# ax3.grid()
# fig.suptitle("Motion Artefact on TROIKA Dataset")
# fig.savefig("./images/artefsTroika.pdf")
# plt.close()

# exit(0)

# data_fls, ref_fls = LoadTroikaDataset()

# means = np.zeros(len(ref_fls))
# stds = np.zeros(len(ref_fls))
# maxs = np.zeros(len(ref_fls))
# mins = np.zeros(len(ref_fls))
# for i in range(len(ref_fls)): # For each file
#     # ref = LoadCapnobaseDataFile(data_fls[i], ref_fls[i])[1][1,:]
#     ref = sp.io.loadmat(ref_fls[i])["BPM0"]
#     means[i] = np.mean(ref)
#     stds[i] = np.std(ref)
#     mins[i] = np.min(ref)
#     maxs[i] = np.max(ref)
# print(means)
# print(stds)
# print(mins)
# print(maxs)
# print(np.mean(means))
# print(np.max(maxs))
# print(np.min(mins))
# exit(0)


#### TROIKA Single Window Visualisation ###

dataset = "troika"
data_fls, ref_fls = LoadTroikaDataset()

# CHOOSE FILE
file_num = 3

data_fl, ref_fl = list(zip(data_fls, ref_fls))[file_num:file_num+1][0]

# Load data using LoadTroikaDataFile
Fs = 125 # Troika data has sampling rate of 125 Hz
ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
# print(data_fl)
ref = sp.io.loadmat(ref_fl)

winSize = 8*Fs # Ground truth BPM provided in 8 second windows
winShift = 2*Fs # Successive ground truth windows overlap by 2 seconds

# CHOOSE WINDOW IN FILE
eval_window_idx = 62

# Choose method
method = "FFT_Peak_Acc"


offset = eval_window_idx*winShift

window_start = offset
window_end = winSize+offset
offset += winShift

print(f"Win start,end: {window_start}, {window_end}")
ppg_window = ppg[window_start:window_end]
accx_window = accx[window_start:window_end]
accy_window = accy[window_start:window_end]
accz_window = accz[window_start:window_end]

ppg_window = ScaleSignal(BandpassFilter(ppg_window, Fs))

# ecg = sp.io.loadmat(data_fl)["sig"][0][window_start:window_end]
# peaks, _ = find_peaks(ecg, prominence=150, distance=66)
# plt.figure()
# plt.plot(ecg)
# plt.plot(peaks, ecg[peaks],"x")
# plt.savefig("./test.pdf")
# plt.close()
# print(60*len(peaks)*Fs/len(ecg))
# print(60*125/np.mean(np.diff(peaks)))
# print(np.mean(60*125/np.diff(peaks)))
# print(np.median(60*125/np.diff(peaks)))
# print(ref['BPM0'][eval_window_idx][0])
# exit(0)

# peaks = find_peaks(ppg_window, **get_find_peaks_args_time(Fs))[0]
# beats = bb.ppgtools.ppg_detectbeats(ppg_window, 125, method="scipy")
# print(beats)
# peaks = bb.ppgtools.ppg_detectpeaks(ppg_window, 125, method="scipy")
# print(peaks["Peak_locs"])


# print(bb.ppgtools.from_cycles(ppg_window,peaks["Peak_locs"], peaks["Trough_locs"], 125)["ppg_PR_mean"])
# print(60*125/np.mean(np.diff(peaks)))
# print(np.mean(60*125/np.diff(peaks)))
# print(np.mean(60*125/np.diff(peaks)))
# print(60*125/np.mean(np.diff(beats)))
# print(ref['BPM0'][eval_window_idx][0])

# plt.figure()
# plt.plot(ppg_window)
# peaks = find_peaks(ppg_window, **get_find_peaks_args_time(Fs))[0]
# plt.plot(peaks, ppg_window[peaks],"x")
# plt.savefig("./test.pdf")
# plt.close()
# exit(0)

pred, conf = AnalyzeWindow(ppg_window, accx_window, accy_window, accz_window, method, Fs=Fs, verbose=True)

groundTruthBPM = ref['BPM0'][eval_window_idx][0]
print('Ground Truth BPM: ', groundTruthBPM)

predError = groundTruthBPM - pred
print("Prediction Error: ", predError)

