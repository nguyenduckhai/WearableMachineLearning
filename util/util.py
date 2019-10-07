import pandas as pd
import numpy as np
from os import listdir
import matplotlib.pyplot as plt
from os.path import isfile, join
from scipy.stats import entropy
import peakutils
from librosa.feature import zero_crossing_rate
from scipy.fftpack import fft
from scipy import signal
from datetime import datetime

mypath = "/home/khai/SENSOR_PROJECT/Data/"


action_idx = {
    "Fall": 0,
    "Wabble" :1,
    "Walk": 2,
    "Stand": 3,
    "Sit" : 4
}

def get_activity(activity_id,df_activities):
    for ind in range(len(df_activities)):
        if (df_activities.iloc[ind,0] == activity_id):
            return df_activities.iloc[ind,1]

def get_duration(data_labels):
    row,col = data_labels.shape 
    output = np.zeros([row,1])
    for ind in range(row): 
        if(data_labels.iloc[ind][5] == True):
            time_start = datetime.strptime(data_labels.iloc[ind][3], '%Y-%m-%d %H:%M:%S.%f')
            time_end = datetime.strptime(data_labels.iloc[ind][4], '%Y-%m-%d %H:%M:%S.%f')
            output[ind,:] = np.abs(time_end.second-time_start.second)
        else: 
            output[ind,:] = 0
    return output



def check_duration(duration,acc_id,gyro_id):
    if(len(acc_id) == duration and len(gyro_id) == duration):
        return True
    return False

def get_acc_id(df_acc,time_start,time_end):
    df_acc.timest = df_acc.timest.astype(str)
    acc_id = []
    for i in range(len(df_acc)):
        if(df_acc.timest.iloc[i] >= time_start and df_acc.timest.iloc[i] <= time_end):
            acc_id.append(i)
    return acc_id

def get_gyro_id(df_gyro,time_start,time_end):
    df_gyro.timest = df_gyro.timest.astype(str)
    gyro_id = []
    for i in range(len(df_gyro)):
        if(df_gyro.timest.iloc[i] >= time_start and df_gyro.timest.iloc[i] <= time_end):
            gyro_id.append(i)
    return gyro_id

def get_time(df_label):
    time_start = np.empty((len(df_label),1),object)
    time_end = np.empty((len(df_label),1),object)  
    for i in range(0,len(df_label)):
        time_start[i,:] = df_label.iloc[i][3]
        time_end[i,:] = df_label.iloc[i][4]
    return time_start,time_end

def get_fft(data,sampling_rate):
    row,col = data.shape
    N = len(data)
    # Nyquist Sampling Criteria
    T = 1/sampling_rate # inverse of the sampling rate
    x = np.linspace(0.0, 1.0/(2.0*T), int(N/2))
    yr = []
    for i in range(col):
        yr.append(fft(data[:,i])) # "raw" FFT with both + and - frequencies
#         y =  2/N * np.abs(yr[0:np.int(N/2)]) # positive freqs only
    return np.asarray(yr)

def spectrum_energy(window,sampling_rate):
    window_freq_raw = get_fft(window,sampling_rate)
    PSD = np.abs(window_freq_raw)**2
    total_energy = np.sum(PSD,1)
    return total_energy

def spectrum_entropy(window,sampling_rate):
    N = len(window)
    window_freq_raw = get_fft(window,sampling_rate)
    PSD_norm = 2/N*np.abs(window_freq_raw)**2
    Entropy = []
    for i in range(len(PSD_norm)):
        p = []
        for z in range(len(PSD_norm[i])):
            p.append(PSD_norm[i][z]/sum(PSD_norm[i]))
        Entropy.append(entropy(p))
    return Entropy

def peak_count(window):
    peaks = []
    for i in range(window.shape[1]):
        # Using lib peakutils for signal to return ind of highest peak
        peak =  np.array(peakutils.indexes(window[:,i],thres=0.02/max(window[:,i]),min_dist=0.1))
        peaks.append(len(peak))
    return peaks
    
def get_zero_crossing_rate(window):
    zcrs = []
    for i in range(window.shape[1]):
        axis = window[:,i]
        # zero_corssing_rate will find ind that cross zero and np.mean it 
        zcr = zero_crossing_rate(axis,frame_length=len(axis),hop_length=len(axis),center=False)
        # Because it return np.array
        zcrs.append(zcr[0][0])
    return zcrs

def get_data(matrix_data,y,chunk):
    # Chunk the matrix list into 10 array list 
    window_list = list(chunks(np.arange(0, len(matrix_data)), chunk))
    X = []
    for i in range(len(window_list)):
        # get data for each window from matrix data
        window = np.array(matrix_data[window_list[i][0]:window_list[i][-1]+1])
        # Calculate the mean of window
        mean = np.mean(window,axis=0)
        # calculate the std of window 
        std = np.std(window,axis=0)
        # Count the peak 
        peaks = peak_count(window)
        # Zero crossing rate of each window 
        zcr = get_zero_crossing_rate(window)
        # get spectrum_energy
        total_energy = spectrum_energy(window,10)
        # get spectrum_entropy
        se = spectrum_entropy(window,10)
        # Put in to a np.array
        X.append(np.concatenate([mean,std,peaks,zcr,total_energy,se]))
    return np.asarray(X), np.full((len(window_list),1),y)

def get_data_second_way(matrix_data,y,chunk):
    # Chunk the matrix list into 10 array list 
    window_list = list(chunks(np.arange(0, len(matrix_data)), chunk))
    X = []
    for i in range(len(window_list)):
        # get data for each window from matrix data
        window = np.array(matrix_data[window_list[i][0]:window_list[i][-1]+1])
        # Calculate the mean of window
        mean = np.mean(window,axis=0)
        # calculate the std of window 
        std = np.std(window,axis=0)
        # Count the peak 
        # peaks = peak_count(window)
        # Zero crossing rate of each window 
        zcr = get_zero_crossing_rate(window)
        # get spectrum_energy
        total_energy = spectrum_energy(window,10)
        # get spectrum_entropy
        se = spectrum_entropy(window,10)
        # Put in to a np.array
        X.append(np.concatenate([mean,std,peaks,zcr,total_energy,se]))
    return np.asarray(X), np.full((len(window_list),1),y)

# def load_data(fpath):
#     data = pd.read_csv(fpath,header = None, names = None)
#     filename = fpath[fpath.rfind("/")+1:]
#     data_list = []
#     for idx in range(len(data[2])):
#         data_list.append(list(data[2][idx].split(","))) 
#     columns = ['device1','acc_x','acc_y','acc_z','device2','gyro_x','gyro_y','gyro_z','data','time','Longtitude','Latitude']
#     df = pd.DataFrame(data_list,columns=columns)
#     matrix_data = df[['acc_x','acc_y','acc_z','gyro_x','gyro_y','gyro_z']]
#     y = find_action(filename)
#     return matrix_data,y

def load_data(df_label,df_acc,df_gyro,chunk):
    durations = get_duration(df_label)
    time_start,time_end = get_time(df_label)
    X = np.empty((0,36), float)
    y = np.empty((0,1),float)
    for i in range(len(df_label)):
        if(durations[i] != 0):
            activity_id = df_label.activity_id.iloc[i]
            acc_id = get_acc_id(df_acc,time_start[i],time_end[i])
            gyro_id = get_gyro_id(df_gyro,time_start[i],time_end[i])
            # if(durations[i] == len(acc_id)/10 and durations[i] == len(gyro_id)/10):
            if(len(acc_id) == len(gyro_id)):
                # matrix_data = np.concatenate([df_acc.iloc[acc_id,2:5],df_gyro.iloc[gyro_id,2:5]],axis=1)
                matrix_data = np.concatenate([df_acc[['x_value','y_value','z_value']].iloc[acc_id],
                                              df_gyro[['x_value','y_value','z_value']].iloc[gyro_id]],axis=1)
                X_input,y_input = get_data(matrix_data,activity_id,chunk)
                X = np.append(X,X_input,axis=0)
                y = np.append(y,y_input,axis=0)
    return X,y

def ChangeStr_toFloat(df):
    df.acc_x = df.acc_x.astype(float)
    df.acc_y = df.acc_y.astype(float)
    df.acc_z = df.acc_y.astype(float)
    df.gyro_x = df.acc_x.astype(float)
    df.gyro_y = df.acc_y.astype(float)
    df.gyro_z = df.acc_y.astype(float)
    return df

def find_action(filename):
    split_array = filename.split("_")
    for i in split_array:
        if i == "Wabble":
            return action_idx["Wabble"]
        elif i == "Fall": 
            return action_idx["Fall"]
        elif i == "Walk": 
            return action_idx["Walk"]
        elif i == "Stand": 
            return action_idx["Stand"]
        elif i == "Sit": 
            return action_idx["Sit"]
        elif i == "Fall2": 
            return action_idx["Fall"]
        elif i == "Fall3": 
            return action_idx["Fall"]
         
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# def add_data_toX(fdir):
#     files = [f for f in listdir(fdir) if isfile(join(fdir, f))]
#     X = np.empty((0,24), float)
#     y = np.empty((0,1),float)
#     for file in files:
#         matrix_data,y_output = load_data(fdir + file)
#         matrix_data = ChangeStr_toFloat(matrix_data)
#         X_input,y_input = get_data(matrix_data,y_output)
#         X = np.append(X,X_input,axis=0)
#         y = np.append(y,y_input,axis=0)
#     # return np.asarray(X)
#     return X,y

# ---------------------------------FILTER-----------------------------------------------------------------
def median_filter(data, f_size):
    lgth, num_signal=data.shape
    f_data=np.zeros([lgth, num_signal])
    for i in range(num_signal):
        f_data[:,i]=signal.medfilt(data[:,i], f_size)
    return f_data

def freq_filter(data, f_size, cutoff):
    lgth, num_signal=data.shape
    f_data=np.zeros([lgth, num_signal])
    lpf=signal.firwin(f_size, cutoff, window='hamming')
    for i in range(num_signal):
        f_data[:,i]=signal.convolve(data[:,i], lpf, mode='same')
    return f_data

def set_data_acc(df_acc):
    rows,col = df_acc.shape
    data = np.zeros([rows,3])
    data[:,0] = df_acc.x_value
    data[:,1] = df_acc.y_value
    data[:,2] = df_acc.z_value
    return data

def set_gyro_acc(df_gyro):
    rows,col = df_gyro.shape
    data = np.zeros([rows,3])
    data[:,0] = df_gyro.x_value
    data[:,1] = df_gyro.y_value
    data[:,2] = df_gyro.z_value
    return data

def get_filter_comb_data(df_label,df_acc,df_gyro): 
    fs=10
    cutoff=2
    durations = get_duration(df_label)
    time_start,time_end = get_time(df_label)
    data_acc = set_data_acc(df_acc)
    data_gyro = set_gyro_acc(df_gyro)
    for i in range(len(df_label)):
        if(durations[i] != 0):
            activity_id = df_label.activity_id.iloc[i]
            acc_id = get_acc_id(df_acc,time_start[i],time_end[i])
            gyro_id = get_gyro_id(df_gyro,time_start[i],time_end[i])
            if(durations[i] == len(acc_id)/10 and durations[i] == len(gyro_id)/10):
                median_data_acc=median_filter(data_acc[acc_id], 9)
                comb_data_acc=freq_filter(median_data_acc, 10, cutoff/fs)

                median_data_gyro=median_filter(data_gyro[gyro_id], 9)
                comb_data_gyro=freq_filter(median_data_gyro, 10, cutoff/fs)
                
                df_acc.iloc[acc_id,2:5] = comb_data_acc
                df_gyro.iloc[gyro_id,2:5] = comb_data_gyro
    return df_acc,df_gyro

def get_filter_comb_data_faster(df_label,df_acc,df_gyro): 
    fs=10
    cutoff=2
    durations = get_duration(df_label)
    # time_start,time_end = get_time(df_label)
    data_acc = set_data_acc(df_acc)
    data_gyro = set_gyro_acc(df_gyro)
    for i in range(len(df_label)):
        if(durations[i] != 0):
            id = df_label.id.iloc[i]
            acc_id = df_acc.index[(df_acc.wearable_label_id == id) == True].tolist()
            gyro_id = df_gyro.index[(df_gyro.wearable_label_id == id) == True].tolist()
            if(durations[i] == len(acc_id)/10 and durations[i] == len(gyro_id)/10):
                median_data_acc=median_filter(data_acc[acc_id], 9)
                comb_data_acc=freq_filter(median_data_acc, 10, cutoff/fs)

                median_data_gyro=median_filter(data_gyro[gyro_id], 9)
                comb_data_gyro=freq_filter(median_data_gyro, 10, cutoff/fs)
                
                df_acc.iloc[acc_id,2:5] = comb_data_acc
                # df_acc[['x_value','y_value','z_value']].iloc[acc_id] = comb_data_acc
                df_gyro.iloc[gyro_id,2:5] = comb_data_gyro
                # df_gyro[['x_value','y_value','z_value']].iloc[gyro_id] = comb_data_gyro
    return df_acc,df_gyro


# def get_fFFT provides us spectrum density( i.e. frequency) of the time-domain signal.  So, PSD  is defined taking square the of absolute value of FFT. 
# ro): 
#     fs=10FFT provides us spectrum density( i.e. frequency) of the time-domain signal.  So, PSD  is defined taking square the of absolute value of FFT. 

#     cutoff=2
#     durations = get_duration(df_label)
#     time_start,time_end = get_time(df_label)
#     data_acc = set_data_acc(df_acc)
#     data_gyro = set_gyro_acc(df_gyro)
#     for i in range(len(df_label)):
#         if(durations[i] != 0):
#             activity_id = df_label.activity_id.iloc[i]
#             acc_id = get_acc_id(df_acc,time_start[i],time_end[i])
#             gyro_id = get_gyro_id(df_gyro,time_start[i],time_end[i])
#             if(durations[i] == len(acc_id)/10 and durations[i] == len(gyro_id)/10):
#                 median_data_acc=median_filter(data_acc[acc_id], 9)

#                 median_data_gyro=median_filter(data_gyro[gyro_id], 9)
                
#                 df_acc.iloc[acc_id,2:5] = median_data_acc
#                 df_gyro.iloc[gyro_id,2:5] = median_data_gyro
#     return df_acc,df_gyro

def get_filter_lowpass_filter(df_label,df_acc,df_gyro): 
    fs=10
    cutoff=2
    durations = get_duration(df_label)
    time_start,time_end = get_time(df_label)
    data_acc = set_data_acc(df_acc)
    data_gyro = set_gyro_acc(df_gyro)
    for i in range(len(df_label)):
        if(durations[i] != 0):
            activity_id = df_label.activity_id.iloc[i]
            acc_id = get_acc_id(df_acc,time_start[i],time_end[i])
            gyro_id = get_gyro_id(df_gyro,time_start[i],time_end[i])
            if(durations[i] == len(acc_id)/10 and durations[i] == len(gyro_id)/10):
                low_pass_data_acc = freq_filter(data_acc[acc_id],10,cutoff/fs)

                low_pass_data_gyro = freq_filter(data_gyro[gyro_id],10,cutoff/fs)

                df_acc.iloc[acc_id,2:5] = low_pass_data_acc
                df_gyro.iloc[gyro_id,2:5] = low_pass_data_gyro
    return df_acc,df_gyro


def fft_plot(data, fs, title):
    lgth, num_signal=data.shape
    yr=np.zeros([lgth,num_signal])
    yr[:,0]=fft(data[:,0])
    yr[:,1]=fft(data[:,1])
    yr[:,2]=fft(data[:,2])
    # Nyquist Sampling Criteria
    T = 1/fs # inverse of the sampling rate
    x = np.linspace(0.0, 1.0/(2.0*T), int(lgth/2))
    
    fig, ax=plt.subplots()
    labels=['x','y','z']
    color_map=['r', 'g', 'b']
    for i in range(3):
        # FFT algorithm
        y =  2/lgth * np.abs(yr[0:np.int(lgth/2),i]) # positive freqs only
        ax.plot(x,y, color_map[i], label=labels[i])
    ax.set_xlim([0, fs/2])
    ax.set_xlabel('Hz')
    ax.set_title('Frequency spectrum: '+title) 
    ax.legend()

def plot_lines(data, fs, title):
    num_rows, num_cols=data.shape
    if num_cols!=3:
        raise ValueError('Not 3D data')
    fig, ax=plt.subplots()
    labels=['x','y','z']
    color_map=['r', 'g', 'b']
    index=np.arange(num_rows)/fs
    for i in range(num_cols):
        ax.plot(index, data[:,i], color_map[i], label=labels[i])
    ax.set_xlim([0,num_rows/fs])
    ax.set_xlabel('Time [sec]')
    ax.set_title('Time domain: '+title)
    ax.legend()

# ---------------------------------EXTRA-----------------------------------------------------------------
def load_data_faster(df_label,df_acc,df_gyro,chunk):
    df_acc.x_value = df_acc.x_value.astype(float)
    df_acc.y_value = df_acc.y_value.astype(float)
    df_acc.z_value = df_acc.z_value.astype(float)

    df_gyro.x_value = df_gyro.x_value.astype(float)
    df_gyro.y_value = df_gyro.y_value.astype(float)
    df_gyro.z_value = df_gyro.z_value.astype(float)
    durations = get_duration(df_label)
    X = np.empty((0,36), float)
    y = np.empty((0,1),float)
    for i in range(len(df_label)):
        if(durations[i] != 0):
            id = df_label.id.iloc[i]
            activity_id = df_label.activity_id.iloc[i]
            acc_id = df_acc.index[(df_acc.wearable_label_id == id) == True].tolist()
            gyro_id = df_gyro.index[(df_gyro.wearable_label_id == id) == True].tolist()
            # if(durations[i] == len(acc_id)/10 and durations[i] == len(gyro_id)/10):
            if(len(acc_id) == len(gyro_id)):
                # matrix_data = np.concatenate([df_acc.iloc[acc_id,2:5],df_gyro.iloc[gyro_id,2:5]],axis=1)
                matrix_data = np.concatenate([df_acc[['x_value','y_value','z_value']].iloc[acc_id],
                                              df_gyro[['x_value','y_value','z_value']].iloc[gyro_id]],axis=1)
                X_input,y_input = get_data(matrix_data,activity_id,chunk)
                X = np.append(X,X_input,axis=0)
                y = np.append(y,y_input,axis=0)
    return X,y

def change_wearable_id(df_label,df_acc,df_gyro):
    df_gyro.timest = df_gyro.timest.astype(str)
    df_acc.timest = df_acc.timest.astype(str)
    time_start,time_end = get_time(df_label)
    for j in range(len(time_start)):
        for i in range(len(df_acc)):
            if(df_acc.timest.iloc[i] >= time_start[j] and df_acc.timest.iloc[i] <= time_end[j]):
                df_acc.wearable_label_id.at[i] = j+1
    for j in range(len(time_start)):
        for i in range(len(df_gyro)):
            if(df_gyro.timest.iloc[i] >= time_start[j] and df_gyro.timest.iloc[i] <= time_end[j]):
                df_gyro.wearable_label_id.at[i] = j+1
    return df_acc,df_gyro


def plot_mean(mean):
    X = []
    X = np.asarray(mean)
    cols, rows = X.shape
    labels=['x_acc_mean','y_acc_mean','z_acc_mean','x_gyro_mean','y_gyro_mean','z_gyro_mean']
    for i in range(rows):
        plt.plot(X[:,i])
        plt.title(labels[i])
        plt.ylabel('values')
        plt.xlabel('chunks')
        plt.show()

def plot_std(std):
    X = []
    X = np.asarray(std)
    cols, rows = X.shape
    labels=['x_acc_std','y_acc_std','z_acc_std','x_gyro_std','y_gyro_std','z_gyro_std']
    for i in range(rows):
        plt.plot(X[:,i])
        plt.title(labels[i])
        plt.ylabel('values')
        plt.xlabel('chunks')
        plt.show()

def plot_peaks_count(peak):
    X = []
    X = np.asarray(peak)
    cols, rows = X.shape
    labels=['x_acc_peaks_count','y_acc_peaks_count','z_acc_peaks_count','x_gyro_peaks_count','y_gyro_peaks_count','z_gyro_peaks_count']
    for i in range(rows):
        plt.plot(X[:,i])
        plt.title(labels[i])
        plt.ylabel('peaks')
        plt.xlabel('chunks')
        plt.show()

def plot_zcr(zcr):
    X = []
    X = np.asarray(zcr) 
    cols, rows = X.shape
    labels=['x_acc_zcr','y_acc_zcr','z_acc_zcr','x_gyro_zcr','y_gyro_zcr','z_gyro_zcr']
    for i in range(rows):
        plt.plot(X[:,i])
        plt.title(labels[i])
        plt.ylabel('zcr')
        plt.xlabel('chunks')
        plt.show()

def plot_spectrum_energy(spectrum_energy):
    X = []
    X = np.asarray(spectrum_energy) 
    cols, rows = X.shape
    labels=['x_acc_se','y_acc_se','z_acc_se','x_gyro_se','y_gyro_se','z_gyro_se']
    for i in range(rows):
        plt.plot(X[:,i])
        plt.title(labels[i])
        plt.ylabel('energy')
        plt.xlabel('chunks')
        plt.show()