from scipy.io import wavfile as w
from scipy.fftpack import dct 
import numpy as np
from numpy.fft import rfft as DFT
import matplotlib.pyplot as plt
import parselmouth as pm
 
def delta(arr,n):
    cmn = np.roll(arr,-n,axis=1)
    cpn = np.roll(arr,n,axis=1)
    return cpn-cmn

def hertztomel(a,method='std'):
    # The std formulation is on matlab and librosa
    if method=='std':
        return 2595*np.log10(1+a/700)
    return (1000/np.log10(2))*np.log10(1+a/1000)

def meltohertz(a,method='std'):
    # The std formulation is on matlab and librosa
    if method == 'std':
        return 700*(10**(a/2595)-1)
    return 1000*(10**(a*np.log10(2)/1000)-1)

def mel_filterbank(N,n_filters=13,Fs=12000,min_freq=0,max_freq = None):
    """
    Compute a transformation matirx for mel filterbanks

    Args:
        N: 
            number of DFT bins
        n_filters: 
            number of mel filters in the filterbank
        min_freq:
            Minimum filter frequency in Hz
        max_freq:
            maximum filter frequency in Hz
        Fs:
            Sampling frequency of the signal
    
    Return:
        the transformation matrix of mel-filterbank
        Shape: (n_filters, N//2+1)

    https://github.com/ddbourgin/numpy-ml/blob/b0359af5285fbf9699d64fd5ec059493228af03e/numpy_ml/preprocessing/dsp.py#L276
    """

    # define mel-filter parameters
    max_freq = Fs/2 if max_freq is None else max_freq
    min_mel, max_mel = hertztomel(min_freq), hertztomel(max_freq)

    # Create the empty filterbank transformation matrix
    fbank = np.zeros((n_filters, N//2+1))

    # ensure to scale the melscale to the size of spectrum N//2+1 as the they will be used for indexing
    melscale_bins = np.floor(meltohertz(np.linspace(min_mel,max_mel,n_filters+2))/max_freq*(N//2+1)).astype(int)
    melscale_spaces = np.diff(melscale_bins).astype(int)

    # Print Filterbank details:
    print(f'\n\nFilterbank Details: \nmin_freq = {min_freq}, max_freq = {max_freq}')
    print(f"MEL min = {min_mel}, MEL max = {max_mel}")
    print(f"MEL Bins: {melscale_bins}")
    print(f"MEL BW {melscale_spaces}")

    # # sampling bins of the DFT
    # hzscale_bins = np.linspace(0, Fs/2, N//2+1,endpoint=True)

    # fltrs = melscale_bins.reshape(-1,1)-hzscale_bins.reshape(1,-1)
    # for i in range(n_filters):
    #     # Create the triangular filter at the correct mel frequency and bandwidth
    #     # Calculate the filter value
    #     Nl = -fltrs[i]/melscale_spaces[i]
    #     Nr = fltrs[i+2]/melscale_spaces[i+1]

    #     fbank[i] = np.maximum(0,np.minimum(Nl,Nr))

    for n in range(n_filters):
        fbank[n,melscale_bins[n]:melscale_bins[n+1]] = np.linspace(0,1,melscale_spaces[n])
        fbank[n,melscale_bins[n+1]:melscale_bins[n+2]] = np.linspace(1,0, melscale_spaces[n+1])

    # Area normalization of triangular filters
    enr_norm = 2/(melscale_bins[2:n_filters+2]-melscale_bins[:n_filters])
    fbank *= enr_norm[:,np.newaxis] 
    return fbank

def Extract_Feature_Vec(link):

    # Get MFCC Vectors
        # window_length = 0.015 secs
        # jump = 0.005 secs
    
    # load signal
    Fs, sig = w.read(link)
    N = len(sig)

    # window the signal
    win_len = int(Fs*0.015)
    win_jump = int(Fs*0.005)

    print(f'Computing MFCCs: \nWindow Length: {win_len}\nWindow Jump: {win_jump}\nNumber of Sample: {N}\nSampling Frequency: {Fs} Hz')
    win_sig = [sig[i:i+win_len] for i in range(0,N-win_len,win_jump)]
    num_wins = len(win_sig)
    print(f'Number of windows: {num_wins}')
    # Find energy of the signal
    E_sig = [i**2 for i in win_sig]
    E_sig = np.array(E_sig)
    dE_sig = delta(E_sig,2)
    ddE_sig = delta(dE_sig,1)

    # Find MFCCs of the windowed signal
        # compute the power spectrum of the signals(DFT**2)
        # apply the filterbank on each windowed power sepctrum
        # multiply each filterbank spectrum with power spectrum 
    power_spec_sig = np.array([np.abs(DFT(s)/len(s))**2 for s in win_sig])
    fbank = mel_filterbank(win_len,Fs=Fs)
    mel_spectrogram = np.dot(fbank, power_spec_sig.T)
  
    # compute the MFCCs by logging the mel spectrogram and computing the DCT
    mfccs = [dct(np.log10(i),axis=0,norm='ortho') for i in mel_spectrogram]
    mfccs = np.array(mfccs)
    # find delta MFCCs and delta delta mfccs
    dmfccs = delta(mfccs,2)
    ddmfccs = delta(dmfccs,1)

    # Net energies in each window
    E_sig = np.sum(E_sig,axis=1)
    dE_sig = np.sum(dE_sig,axis=1)
    ddE_sig = np.sum(ddE_sig,axis=1)

    print(f'MFCCs: \n{mfccs.shape}\n{dmfccs.shape}\n{ddmfccs.shape}')
    print(f'Energy: \n{E_sig.shape}\n{dE_sig.shape}\n{ddE_sig.shape}')
    
    # combine all and return the final feature vectors
    MFCCS_vector = np.zeros((42,num_wins))
    MFCCS_vector[:39,:] = np.vstack((mfccs,dmfccs,ddmfccs))
    MFCCS_vector[39:,:] = np.vstack((E_sig,dE_sig,ddE_sig))

    print(f"MFCC Feature Vectors: {MFCCS_vector.shape}")

    # cepstral mean Substraction
    for i in range(MFCCS_vector.shape[0]):
        MFCCS_vector[i] -= np.mean(MFCCS_vector[i])

    return MFCCS_vector

def MFCC_vecs_praat(path):
    sound = pm.Sound(path)

    # Find MFCC
    mfcc_obj = sound.to_mfcc(number_of_coefficients=13)
    mfcc = mfcc_obj.to_array()
    mfcc[0] = mfcc[0]**2                # configure c0 for energy
    dmfcc = delta(mfcc,2) 
    ddmfcc = delta(dmfcc,1)
    print(f"\n\nPraat MFCCs: {mfcc.shape}, {dmfcc.shape}, {ddmfcc.shape}")

    # Combine all vectors:
    MFCC_vecs = np.vstack((mfcc[1:],dmfcc[1:],ddmfcc[1:],mfcc[0],dmfcc[0],ddmfcc[0]))
    print(f"Final MFCC Vector Shape: {MFCC_vecs.shape}")
    return MFCC_vecs


if __name__ == '__main__':
    sample_link = './Data/Dev/Team-05/111a.wav'
    fv = Extract_Feature_Vec(sample_link)
    fs, sig = w.read(sample_link)
    filters = mel_filterbank(300,Fs=fs)
    p_fv = MFCC_vecs_praat(sample_link)

    # Data presentation:
    fig,ax = plt.subplots(2,2,figsize=(15,15))
    ax[0,0].imshow(filters,aspect='auto',origin='lower')
    ax[0,0].set_title('Mel filterbank')

    ax[0,1].imshow(fv,aspect='auto',origin='lower')
    ax[0,1].set_title('Mel Spectrogram')

    for n in range(filters.shape[0]):
        ax[1,0].plot(filters[n])
    ax[1,0].set_title('Mel Filters')

    ax[1,1].plot(sig)
    ax[1,1].set_title('Sample Signal')

    # Data Presentation for praat
    plt.figure(figsize=(10,10))
    plt.imshow(p_fv,aspect='auto',origin='lower')
    plt.title('Mel Spectogram using Praat(parselmouth)')
    plt.show()
   


