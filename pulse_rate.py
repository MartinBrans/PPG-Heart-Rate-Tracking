
# ## Part 1: Pulse Rate Algorithm
# 
# ### Dataset
# **Troika**[1] dataset is used to develop the motion-compensated pulse rate algorithm.
# 
# 1. Zhilin Zhang, Zhouyue Pi, Benyuan Liu, ‘‘TROIKA: A General Framework for Heart Rate Monitoring Using Wrist-Type Photoplethysmographic Signals During Intensive Physical Exercise,’’IEEE Trans. on Biomedical Engineering, vol. 62, no. 2, pp. 522-531, February 2015. Link
# 
# ### Capnobase
# W. Karlen, S. Raman, J. M. Ansermino and G. A. Dumont, "Multiparameter Respiratory Rate Estimation From the Photoplethysmogram," in IEEE Transactions on Biomedical Engineering, vol. 60, no. 7, pp. 1946-1953, July 2013, doi: 10.1109/TBME.2013.2246160.
# -----


# https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_peak_finding.py#L729-L1010

# ### Code


import glob

import numpy as np
import scipy as sp
import scipy.io

import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd
import random
import math
import ctypes

import heartpy as hp

### Parameters begin ###

# Uncomment to turn warnings into errors
# import warnings
# warnings.simplefilter("error")

# dataset = "capnobase"
dataset = "troika"
plotResults = False

# Algo parameters
filter_on = True # False to disable the scipy butterworth order 3 bandpass filter on ppg and acc signals
minBeats = 5 # For algorithms based on signal peak detection. Assuming 40 BPM or lower is impossible, 8 seconds window -> 5 beats at least

# Peak detection parameters for scipy find_peaks on ppg time-based signal:
# findPeaksArgsTime = {"height":10, "distance":35} # Parameters of the original github code by KJStrand (troika dataset)
findPeaksArgsTime = {"prominence":6} # Fiddle for better perf on troika
# findPeaksArgsTime = {"prominence":3} # Parameters that seem good for capnobase dataset

# Peak detection parameters for scipy find_peaks on frequency signals (see FFT_Peak method)
# Actually, KJStrand's implementation normalises the signal before applying fft, so probably no need for tweaks here
findPeaksArgsFreq = {"height":30, "distance":1} # Parameters of the original github code by KJStrad (troika dataset)

### Parameters end ###


# Interfacing with C language, in particular the example algorithm given by Maxim when buying the MAX30101WING
# https://pgi-jcns.fz-juelich.de/portal/pages/using-c-from-python.html
_algo_HR = ctypes.CDLL("./libalgo_HR_flt.so")
# _algo_HR = ctypes.CDLL("./libalgo_HR.so")
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
    data_fls = sorted(glob.glob(data_dir + "/*_signal.csv"))
    ref_fls = sorted(glob.glob(data_dir + "/*_reference.csv"))
    return data_fls, ref_fls

def LoadCapnobaseDataFile(data_fl,ref_fl):
    """
    Load the signal and the reference HR from a Capnobase file (an entry from LoadCapnobaseDataset output list)
    
    Returns: two numpy arrays, the first is the signal, the second is the reference
    """
    data = pd.read_csv(data_fl)["pleth_y"].to_numpy()
    # ref = np.array(pd.read_csv(ref_fl)["hr_ecg_y"].to_numpy()[0].split(" ")[1:],dtype=np.float64)
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
    percentile90_confidence = np.percentile(confidence_est, 10)

    # Find the errors of the best pulse rate estimates
    best_estimates = pr_errors[confidence_est >= percentile90_confidence]
    best_truths = gnd_truths[confidence_est >= percentile90_confidence]

    # return np.mean(np.abs(best_estimates)) # Mean absolute error
    # return np.sqrt(np.mean(best_estimates**2)) # RMS
    # return np.mean(100*np.abs(best_estimates)/best_truths) # in percents
    return (np.mean(np.abs(best_estimates)),np.mean(np.abs(best_estimates)/best_truths)) # both
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
    errsPercent = errs * 100 / trths
    percentile90_confidence = np.percentile(confs, 10)
    errMean = np.mean(errs[confs >= percentile90_confidence])
    errSTD = np.std(errs[confs >= percentile90_confidence])
    # Method sanity check -> ideally the mean error is close to 0 if the algorithm is unbiased (actually it doesn't work well)
    # print(f"Mean error: {np.mean(errs):.3f} BPM")

    # Plots
    if plotResults :
        plt.figure() # Error BPM vs ground truth BPM
        plt.plot(trths[confs >= percentile90_confidence], errs[confs >= percentile90_confidence], '.b')
        plt.plot(trths[confs < percentile90_confidence], errs[confs < percentile90_confidence], 'xr')
        plt.hlines([errMean-1.96*errSTD,errMean,errMean+1.96*errSTD],xmin=60,xmax=180,colors="g",linestyles="dashed")
        plt.grid()
        plt.xlabel("Ground truth HR [BPM]")
        plt.ylabel("Error [BPM]")
        plt.savefig(f"./images/Err_{method}.pdf")
        plt.close()
        
        plt.figure() # Error % vs ground truth BPM
        plt.plot(trths[confs >= percentile90_confidence], errsPercent[confs >= percentile90_confidence], '.b')
        plt.plot(trths[confs < percentile90_confidence], errsPercent[confs < percentile90_confidence], 'xr')
        plt.grid()
        plt.xlabel("Ground truth HR [BPM]")
        plt.ylabel("Error [%]")
        plt.savefig(f"./images/PErr_{method}.pdf")
        plt.close()

        plt.figure() # Histogram error
        plt.hist(errs,20)
        plt.grid()
        plt.xlabel("Error [BPM]")
        plt.ylabel("Occurences")
        plt.savefig(f"./images/Hist_{method}.pdf")
        plt.close()

        plt.figure() # Zoomed histogram error
        plt.hist(errs[np.abs(errs) < 30],20)
        plt.grid()
        plt.xlabel("Error [BPM]")
        plt.ylabel("Occurences")
        plt.savefig(f"./images/ZHist_{method}.pdf")
        plt.close()

        plt.figure() # Histogram error %
        plt.hist(errsPercent,20)
        plt.grid()
        plt.xlabel("Error [%]")
        plt.ylabel("Occurences")
        plt.savefig(f"./images/PHist_{method}.pdf")
        plt.close()

        plt.figure() # Zommed histogram error %
        plt.hist(errsPercent[np.abs(errsPercent	) < 30],20)
        plt.grid()
        plt.xlabel("Error [%]")
        plt.ylabel("Occurences")
        plt.savefig(f"./images/PZHist_{method}.pdf")
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
    # Load data using LoadTroikaDataFile
    
    global dataset
    if dataset == "troika" :
        Fs = 125 # Troika data has sampling rate of 125 Hz, Capnobase 300 Hz
        ppg, accx, accy, accz = LoadTroikaDataFile(data_fl)
        ref = sp.io.loadmat(ref_fl)
        
    elif dataset == "capnobase" :
        Fs = 300
        ppg, ref = LoadCapnobaseDataFile(data_fl, ref_fl)
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
        nWindows = len(ppg)//winSize
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
        except RuntimeWarning: # Heartpy being annoying
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
        except:
            print("Other error")
            print(data_fl)
            print(eval_window_idx)
            pred = 0
            conf = -3e6
        
        if dataset == "capnobase":
            refWindow = ref[1,(ref[0,:] >= window_start/Fs) & (ref[0,:] < window_end/Fs)]
            if len(refWindow) == 0 : continue
            groundTruthBPM = np.mean(refWindow)
        elif dataset == "troika":
            groundTruthBPM = ref['BPM0'][eval_window_idx][0]
        else :
            print("Unknown dataset")
            exit(1)

        if verbose:
            print('Ground Truth BPM: ', groundTruthBPM)
        # if groundTruthBPM < minBPM : minBPM = groundTruthBPM
        # if groundTruthBPM > maxBPM : maxBPM = groundTruthBPM

        predError = groundTruthBPM - pred
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
    
    global findPeaksArgsTime
    global findPeaksArgsFreq
    global filter_on
    if filter_on:
        ppg_bandpass = BandpassFilter(ppg, fs=Fs)
        accx_bandpass = BandpassFilter(accx, fs=Fs)
        accy_bandpass = BandpassFilter(accy, fs=Fs)
        accz_bandpass = BandpassFilter(accz, fs=Fs)
    else:
        ppg_bandpass = ppg
        accx_bandpass = accx
        accy_bandpass = accy
        accz_bandpass = accz

    # Aggregate accelerometer data into single signal
    
    accy_mean = accy-np.mean(accy_bandpass) # Center Y values
    acc_mag_unfiltered = np.sqrt(accx_bandpass**2+accy_mean**2+accz_bandpass**2)
    acc_mag = BandpassFilter(acc_mag_unfiltered, fs=Fs)
    
    prediction = 0
    confidence = 0

    if method == "FFT_Peak":

        peaks = find_peaks(ppg_bandpass, **findPeaksArgsTime)[0]
    
        if verbose:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
            ax1.title.set_text('Signal with Time Domain FindPeaks()')
            ax1.plot(ppg_bandpass)
            ax1.plot(peaks, ppg_bandpass[peaks], "x")
            
            ax2.title.set_text('Aggregated Accelerometer Data')
            ax2.plot(acc_mag, color="purple")
            # plt.show()
            plt.savefig("./images/plot.png")
            
        # Use FFT length larger than the input signal size for higher spectral resolution.
        fft_len=len(ppg_bandpass)*4

        # Create an array of frequency bins
        freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031

        # The frequencies between 40 BPM and 240 BPM Hz
        low_freqs = (freqs >= (40/60)) & (freqs <= (240/60))
        
        mag_freq_ppg, fft_ppg = FreqTransform(ppg_bandpass, freqs, low_freqs, fft_len)
        mag_freq_acc, _ = FreqTransform(acc_mag, freqs, low_freqs, fft_len)
        
        peaks_ppg = find_peaks(mag_freq_ppg, **findPeaksArgsFreq)[0]
        peaks_acc = find_peaks(mag_freq_acc, **findPeaksArgsFreq)[0]
        
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True)
        sorted_freq_peaks_acc = sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True)
        
        # Use the frequency peak with the highest magnitude.
        if len(sorted_freq_peaks_ppg) > 0 :
            use_peak = sorted_freq_peaks_ppg[0]
            # With this method, we do not use acc signal (with the next one we do)
            # for i in range(len(sorted_freq_peaks_ppg)):
            #     # Check nearest two peaks also
            #     cond1 = sorted_freq_peaks_ppg[i] in sorted_freq_peaks_acc
            #     cond2 = sorted_freq_peaks_ppg[i]-1 in sorted_freq_peaks_acc
            #     cond3 = sorted_freq_peaks_ppg[i]+1 in sorted_freq_peaks_acc
            #     if cond1 or cond2 or cond3:
            #         continue
            #     else:
            #         use_peak = sorted_freq_peaks_ppg[i]
            #         break

            chosen_freq = freqs[low_freqs][use_peak]
            prediction = chosen_freq * 60
            confidence = CalcConfidence(chosen_freq, freqs, fft_ppg)
        else :
            print("No freq")
            use_peak = 0
            chosen_freq = 0
        
        if verbose:
            plt.title("PPG Frequency Magnitude")
            plt.plot(mag_freq_ppg)
            plt.plot(peaks_ppg, mag_freq_ppg[peaks_ppg], "x")
            plt.show()
            
            plt.title("ACC Frequency Magnitude")
            plt.plot(mag_freq_acc, color="purple")
            plt.plot(peaks_acc, mag_freq_acc[peaks_acc], "x")
            plt.show()
            
            print("PPG Freq Peaks: ", peaks_ppg)
            print("ACC Freq Peaks: ", peaks_acc)
            
            print("PPG Freq Peaks Sorted: ", sorted_freq_peaks_ppg)
            print("ACC Freq Peaks Sorted: ", sorted_freq_peaks_acc)
            print("Use peak: ", use_peak)
            print(f"Predicted BPM: {prediction}, {chosen_freq} (Hz), Confidence: {confidence}")

    elif method == "FFT_Peak_Acc":

        peaks = find_peaks(ppg_bandpass, **findPeaksArgsTime)[0]
    
        if verbose:
            _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,4))
            ax1.title.set_text('Signal with Time Domain FindPeaks()')
            ax1.plot(ppg_bandpass)
            ax1.plot(peaks, ppg_bandpass[peaks], "x")
            
            ax2.title.set_text('Aggregated Accelerometer Data')
            ax2.plot(acc_mag, color="purple")
            plt.show()
            
        # Use FFT length larger than the input signal size for higher spectral resolution.
        fft_len=len(ppg_bandpass)*4

        # Create an array of frequency bins
        freqs = np.fft.rfftfreq(fft_len, 1 / Fs) # bins of width 0.12207031

        # The frequencies between 40 BPM and 240 BPM Hz
        low_freqs = (freqs >= (40/60)) & (freqs <= (240/60))
        
        mag_freq_ppg, fft_ppg = FreqTransform(ppg_bandpass, freqs, low_freqs, fft_len)
        mag_freq_acc, _ = FreqTransform(acc_mag, freqs, low_freqs, fft_len)
        
        peaks_ppg = find_peaks(mag_freq_ppg, **findPeaksArgsFreq)[0]
        peaks_acc = find_peaks(mag_freq_acc, **findPeaksArgsFreq)[0]
        
        # Sort peaks in order of peak magnitude
        sorted_freq_peaks_ppg = sorted(peaks_ppg, key=lambda i:mag_freq_ppg[i], reverse=True)
        sorted_freq_peaks_acc = sorted(peaks_acc, key=lambda i:mag_freq_acc[i], reverse=True)
        
        # Use the frequency peak with the highest magnitude, unless the peak is also present in the accelerometer peaks.
        if len(sorted_freq_peaks_ppg) > 0 :
            use_peak = sorted_freq_peaks_ppg[0]
            for i in range(len(sorted_freq_peaks_ppg)):
                # Check nearest two peaks also
                cond1 = sorted_freq_peaks_ppg[i] in sorted_freq_peaks_acc
                cond2 = sorted_freq_peaks_ppg[i]-1 in sorted_freq_peaks_acc
                cond3 = sorted_freq_peaks_ppg[i]+1 in sorted_freq_peaks_acc
                if cond1 or cond2 or cond3:
                    continue
                else:
                    use_peak = sorted_freq_peaks_ppg[i]
                    break

            chosen_freq = freqs[low_freqs][use_peak]
            prediction = chosen_freq * 60
            confidence = CalcConfidence(chosen_freq, freqs, fft_ppg)
        else :
            print("No freq")
            use_peak = 0
            chosen_freq = 0
        
        if verbose:
            plt.title("PPG Frequency Magnitude")
            plt.plot(mag_freq_ppg)
            plt.plot(peaks_ppg, mag_freq_ppg[peaks_ppg], "x")
            plt.show()
            
            plt.title("ACC Frequency Magnitude")
            plt.plot(mag_freq_acc, color="purple")
            plt.plot(peaks_acc, mag_freq_acc[peaks_acc], "x")
            plt.show()
            
            print("PPG Freq Peaks: ", peaks_ppg)
            print("ACC Freq Peaks: ", peaks_acc)
            
            print("PPG Freq Peaks Sorted: ", sorted_freq_peaks_ppg)
            print("ACC Freq Peaks Sorted: ", sorted_freq_peaks_acc)
            print("Use peak: ", use_peak)
            print(f"Predicted BPM: {prediction}, {chosen_freq} (Hz), Confidence: {confidence}")      

    elif method == "Peaktime_diff":
        peaks = find_peaks(ppg_bandpass, **findPeaksArgsTime)[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = np.mean(Fs/np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1 # more intervals means more confidence -> actually punishes low BPMs for no reason
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "Peaktime_diff_PostMean":
        peaks = find_peaks(ppg_bandpass, **findPeaksArgsTime)[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.mean(np.diff(peaks)) * 60 # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "Peaktime_diff_Median":
        peaks = find_peaks(ppg_bandpass, **findPeaksArgsTime)[0] # Peaks are expressed as indices
        if len(peaks) < minBeats : 
            # print("insufficient peaks")
            prediction = 0
            confidence = 0
        else :
            prediction = Fs/np.median(np.diff(peaks)) * 60
            confidence = 1
            # confidence = 1 - 1/len(peaks) # more intervals means more confidence -> actually punishes low BPMs for no reason, so do not use this

    elif method == "2thresh":
        lowerThreshold = 0.0
        upperThreshold = np.max(ppg_bandpass)*0.5
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = 0
        nBeats = 0
        for i in range(len(ppg_bandpass)) :
            if ppg_bandpass[i] < lowerThreshold :
                if beatComplete :
                    prediction += Fs / (i-lastTime) * 60
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_bandpass[i] > upperThreshold and beatStarted :
                beatComplete = True
        if nBeats > minBeats : 
            prediction /= nBeats # Outliers could influence the mean -> use median instead?
            confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        else : 
            # print("No beats")
            prediction = 0
            confidence = 0.0

    elif method == "2thresh_PostMean":
        lowerThreshold = 0.0
        upperThreshold = np.max(ppg_bandpass)*0.5
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = 0
        nBeats = 0
        for i in range(len(ppg_bandpass)) :
            if ppg_bandpass[i] < lowerThreshold :
                if beatComplete :
                    prediction += i-lastTime
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_bandpass[i] > upperThreshold and beatStarted :
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
        lowerThreshold = 0.0
        upperThreshold = np.max(ppg_bandpass)*0.5
        beatStarted = False
        beatComplete = False
        lastTime = 0
        prediction = []
        nBeats = 0
        for i in range(len(ppg_bandpass)) :
            if ppg_bandpass[i] < lowerThreshold :
                if beatComplete :
                    prediction.append(i-lastTime)
                    beatComplete = False
                    beatStarted = False
                    nBeats += 1
                if not beatStarted : 
                    lastTime = i
                    beatStarted = True
            if ppg_bandpass[i] > upperThreshold and beatStarted :
                beatComplete = True
        if nBeats > minBeats : 
            prediction = Fs / np.median(prediction) * 60
            confidence = 1
            # confidence = 1 - 1/(nBeats+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        else : 
            # print("No beats")
            prediction = 0
            confidence = 0.0

    elif method == "NumberOfPeaks":
        peaks = find_peaks(ppg_bandpass, **findPeaksArgsTime)[0]
        prediction = len(peaks) / (len(ppg_bandpass)/Fs) * 60
        confidence = 1
        # confidence = 1 - 1/(len(peaks)+1) # more beats means more confidence -> actually punishes low BPMs for no reason, so do not use this
        if len(peaks) == 0 : 
            prediction = 0
            confidence = 0
            # print("No peaks")

    elif method == "MAX FTHR":
        
        ppg_bp_max = np.max(ppg)
        ppg_bp_min = np.min(ppg)

        if ppg_bp_max == ppg_bp_min :
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
            compSpO2 = ctypes.c_ushort(0)

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

            scaleFactor = 2**16 / (ppg_bp_max-ppg_bp_min)
            shift = (ppg_bp_max+ppg_bp_min)/2

            for i in range(len(ppg)):
                
                # temp_ppg_bandpass = int((ppg_bandpass[i] - shift)*scaleFactor) + 2**31
                temp_ppg_bandpass = int((ppg[i] - shift)*scaleFactor + 2**22)

                dinIR.value = temp_ppg_bandpass
                dinRed.value = temp_ppg_bandpass
                dinGreen.value = temp_ppg_bandpass
                ns.value = i

                HRSpO2Func(dinIR,dinRed,dinGreen,ns,SampRate,compSpO2,
                        ir_ac_comp,red_ac_comp,green_ac_comp,ir_ac_mag,red_ac_mag, 
                        green_ac_mag,HRbpm2,SpO2B,DRdy)
                
                if DRdy.value == 1 :
                    prediction += HRbpm2.value#*Fs/100
                    nPredictions += 1

            if nPredictions > 0 :
                prediction /= nPredictions
                confidence = 1 - 1/(nPredictions+1) # more predictions means more confidence -> actually punishes low BPMs for no reason, so do not use this
            else :
                prediction = 0
                confidence = 0
                # print("No predictions")

    elif method == "MAX FTHR Median":
        
        ppg_bp_max = np.max(ppg)
        ppg_bp_min = np.min(ppg)

        if ppg_bp_max == ppg_bp_min :
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
            compSpO2 = ctypes.c_ushort(0)

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

            scaleFactor = 2**16 / (ppg_bp_max-ppg_bp_min)
            shift = (ppg_bp_max+ppg_bp_min)/2

            for i in range(len(ppg)):
                
                # temp_ppg_bandpass = int((ppg_bandpass[i] - shift)*scaleFactor) + 2**31
                temp_ppg_bandpass = int((ppg[i] - shift)*scaleFactor + 2**22)

                dinIR.value = temp_ppg_bandpass
                dinRed.value = temp_ppg_bandpass
                dinGreen.value = temp_ppg_bandpass
                ns.value = i

                HRSpO2Func(dinIR,dinRed,dinGreen,ns,SampRate,compSpO2,
                        ir_ac_comp,red_ac_comp,green_ac_comp,ir_ac_mag,red_ac_mag, 
                        green_ac_mag,HRbpm2,SpO2B,DRdy)
                
                if DRdy.value == 1 :
                    # prediction += HRbpm2.value
                    prediction.append(HRbpm2.value)#*Fs/100)
                    nPredictions += 1

            if nPredictions > 0 :
                prediction = np.median(prediction)
                confidence = 1 - 1/(nPredictions+1) # more predictions means more confidence -> actually punishes low BPMs for no reason, so do not use this
                # confidence = 1
            else :
                prediction = 0
                confidence = 0
                # print("No predictions")

    elif method == "HeartPy" :

        hp_preproc_ppg = hp.scale_data(ppg_bandpass)
        # hp_preproc_ppg = hp.hampel_correcter(ppg,Fs)
        # hp_preproc_ppg = hp.filter_signal(hp_preproc_ppg,[0.5, 6],sample_rate=Fs,order=3,filtertype="bandpass")
        # hp_preproc_ppg = hp.enhance_peaks(hp_preproc_ppg,iterations=2)


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
        

    elif method == "random":
        prediction = random.uniform(40,240)
        confidence = 1

    else :
        print("Unknown method")
        exit(1)

        
    return (prediction, confidence)

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
    norm_x = (x - np.mean(x))/(max(x)-min(x))
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


# Test visualisation of capsobase dataset
# data, ref = LoadCapnobaseDataFile("./datasets/capnobase/data/0009_8min_signal.csv","./datasets/capnobase/data/0009_8min_reference.csv")
# print(data.shape)
# print(ref.shape)
# plt.figure()
# peaks = find_peaks(data[2400:4800],**findPeaksArgsTime)[0]
# plt.plot(data[2400:4800])
# plt.plot(peaks,data[peaks+2400],"xg")
# plt.savefig("./datatest.pdf")
# plt.figure()
# plt.plot(ref[100:150])
# plt.savefig("./reftest.pdf")
# exit(0)

method = "FFT_Peak" # Apply FFT on the window and select most present frequency, convert this frequency to a BPM
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "FFT_Peak_Acc" # (Original TROIKA code) Apply FFT and select most present frequency (+ remove matches with Accelerometer FFT), convert it to a BPM
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "Peaktime_diff" # Compute BPM values based on the time difference between two consecutive peaks, take the mean of the BPM values
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "Peaktime_diff_PostMean" # Compute the mean time difference between peak occurences and convert it to a BPM
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "Peaktime_diff_Median" # Compute the median time difference between peak occurences and convert it to a BPM
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "2thresh" # Compute BPM values based on the duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again), take the mean of the BPM values
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "2thresh_PostMean" # Compute the mean duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again) and convert it to a BPM
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "2thresh_Median" # Compute the median duration of a beat (one beat = hitting an lower threshold, then an upper, then the lower again) and convert it to a BPM
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "NumberOfPeaks" # Compute the number of peaks in the window and convert it to a BPM using the length of the window
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "MAX FTHR" # Base algorithm on MAX32630 FTHR, with FIR filter, finger off detection and peaktime_diff
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "MAX FTHR Median" # Base algorithm on MAX32630 FTHR, with FIR filter, finger off detection and peaktime_diff, take the median of the predictions
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
method = "HeartPy" # HeartPy library
MAE = Evaluate(method)
print(f" Method : {method} =====\n", "MAE is: ", MAE)
# method = "random" # Pick a BPM at random between 40 and 240 BPM
# MAE = Evaluate(method)
# print(f" Method : {method} =====\n", "MAE is: ", MAE)


exit(0)

# **Analyze Single Window of PPG and Accelerometer Data**
# 
# Sample window analysis shown below:


# Single Window Test

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
eval_window_idx = 58

# Choose method
method = "HeartPy"


offset = eval_window_idx*winShift

window_start = offset
window_end = winSize+offset
offset += winShift

print(f"Win start,end: {window_start}, {window_end}")
ppg_window = ppg[window_start:window_end]

accx_window = accx[window_start:window_end]
accy_window = accy[window_start:window_end]
accz_window = accz[window_start:window_end]

# plt.figure()
# plt.plot(ppg_window)
# plt.savefig("./buggy_window.pdf")

pred, conf = AnalyzeWindow(ppg_window, accx_window, accy_window, accz_window, method, Fs=Fs, verbose=False)

groundTruthBPM = ref['BPM0'][eval_window_idx][0]
print('Ground Truth BPM: ', groundTruthBPM)

predError = groundTruthBPM - pred
print("Prediction Error: ", predError)


# **Code Description**
# 
# The Python code may be used in two ways:
# 
# 1) Analyze algorithm performance on Troika dataset.
# 
# The function *RunPulseRateAlgorithm()* expects two files as input:
# - data_fl: .mat file containing PPG and X, Y, Z accelerometer data from Troika dataset (or in Troika format)
# - ref_fl: .mat file containing ground truth heart rates from Troika dataset (or in Troika format)
#        
# RunPulseRateAlgorithm() will compute the heart rate and confidence estimates for 8 second windows of the PPG and accelerometer data, and calculate the error (difference) between the heart estimates and the ground truth heart rate for each 8 second window provided in the reference file.
# 
# The function *Evaluate()* will calculate an overall mean absolute error at 90% availability (see Algorithm Performance section below), assuming the Troika data is located at *./datasets/troika/training_data*.
# 
# 2) Predict heart rate from new input PPG and Accelerometer data.
# 
# The function *AnalyzeWindow()* accepts numpy arrays of ppg, and accelerometer data in the x, y, and z axis, along with a sampling rate. A precondition of the function is that the incoming ppg and acceleromter data was sampled at the provided sampling rate and is aligned in time. While the algorithm was only tested with 8 second windows of PPG and accelerometer data and a sampling rate of 125 Hz, the AnalyzeWindow function may be applied to a longer window sizes or alternate sampling rates. AnalyzeWindow() returns a tuple of (BPM prediction, confidence) for the provided window of ppg and accelerometer data using the provided sample rate.
# 
# **Data Description**
# 
# The data used to design and test this algorithm was taken from the [TROIKA](https://ieeexplore.ieee.org/document/6905737) dataset. The dataset includes 12 subjects ages 18 to 35 with simultaneously recorded two-channel wrist-mounted PPG signals, wrist-mounted three-axis accelerometer signals, and one-channel chest-mounted ECG. The subjects ran on a treadmill at increasing rates to range from resting heart rate to intense exercise heart rate. To build a more complete dataset, it would be beneficial to include a greater number of participants that represent the broader population demographics for age and gender. Additionally, it would be useful to record PPG and accelerometer data from more points on the body that could be used in future wearable devices such as the feet, legs, upper arms, and head. This could provide a more complete picture of how accelerometer signals affect PPG data across the body. Alternative exercises could be recorded as well, such as cylcing, swimming, tennis, and weight lifting, each of which may produce different physiological responses or signal patterns.
# 
# **Algorithm Description**
# 
# The heart rate prediction algorithm takes advantage of the physiology of blood flow through the ventricles of the wrist. Light emitted by an LED on the PPG sensor is reflected less when the ventricles contract and more blood is present, and light is reflected more when the blood returns to the heart and fewer red blood cells are present. Arm movement will also affect the blood levels in the wrist, so periodic motion such as an arm swinging back and forth can also be detected by the PPG signal. 
# 
# The algorithm identifies the strongest frequency components of both the PPG signal and the accelerometer signals to determine which frequency to pick as the heart rate. The algorithm follows these stages:
# 
# 1) Apply bandpass filter to PPG and accelerometer signals to filter out frequencies outside of the 40-240 BPM range.
# 
# 2) Aggregate the X, Y, and Z channels of the accelerometer signal into a signal magnitude signal.
# 
# 2) Tranform the time domain PPG and accelerometer signal to magnitude frequency representations by taking the absolute value of their Fast Fourier Transforms.
# 
# 3) Using the frequency representations of PPG and accelerometer signals, find the peaks with the largest magnitudes, and choose one to be the predicted heart rate frequency.
# 
# - If the highest magnitude peak of both signals is different, choose the highest magnitude peak of the the PPG signal as the heart rate frequency prediction.
# - If the highest magnitude peak of both signals is the same, this may mean that the arm swing signal is overpowering the pulse rate, so choose the next highest magnitude peak of the PPG signal as the heart rate frequency prediction.
# - If each of the highest magnitude peaks of the PPG signal are too close to the peaks of the accelerometer signal, the arm swing frequency could be the same as the pulse rate frequency, so use the highest magnitude peak of the PPG as the heart rate frequency prediction, even though the accelerometer signal has the same peak).
# 
# 4) Convert the chosen peak frequency to a final **BPM Prediction**, and calculate a **Confidence Value** for the chosen frequency by computing the ratio of energy concentrated near that frequency compared to the full signal.
# 
# The BPM Prediction and Confidence Value outputs from this algorithm are not gauranteed to be correct. Confidence values are only used to determine which outputs are very poor, i.e. a low confidence value implies very low signal to noise ratio, since only a small amount of energy is concentrated by the predicted peak. High confidence values do not necessarily imply the that algorithm is significantly more correct, only that the peak at that location is responsible for much more of the signal. Common failure modes include when accelerometer data has random movement spikes or when the three channels combine into a non-periodic signal.
# 
# **Algorithm Performance**
# 
# The algorithm performance was evaluated against the TROIKA reference data. All PPG data was compared against ECG "ground truth" BPM for associated time windows. ECG measures electric potentials across the heart and is considered to be a much more reliable method for obtaining pulse rate than PPG since these electrical signals are not susceptible to the same levels of movement-related noise. Using the confidence estimates to compare prediction quality, the bottom 10% of predictions were discarded while the remaining predictions were evaluated against the ground truth. The final calculated performance metric for the dataset at 90% availability was a mean absolute error (**MAE**) of **13.625 BPM**. This performance may be verified by executing the *Evaluate()* function.
# 
# This algorithm may not perform as well at >90% availability, and does not gaurantee that consecutive or overlapping time windows have similar confidence values, so it is possible that prediction confidence could vary greatly over time if this algorithm was applied to a real-time heart rate analysis scenario. This algorithm also assumes that accelerometer motion will largely result from swinging arms during running on a treadmill and will therefore be consistently periodic. This may not be the case for other activities, such as tennis or basketball.


