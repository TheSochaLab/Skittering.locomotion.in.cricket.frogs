import pickle as pik
import numpy as np
import cv2
from vidHandlr import *
from scipy import interpolate
import scipy.signal as signal
from scipy.stats import linregress
from scipy.optimize import newton
from scipy.interpolate import interp1d

def load_model_data(modeldata):
    with open(modeldata, 'rb') as f:
        mdl_ells = pik.load(f)
        mdl_mnts = pik.load(f)
        mdl_cont = pik.load(f)
        origfn = pik.load(f)
        mdl_segs = pik.load(f)
    return mdl_ells, mdl_mnts, mdl_cont, origfn, mdl_segs
        
def load_wat_ell(watelldata):
    with open(watelldata, 'rb') as f:
        ells = pik.load(f)
        mnts = pik.load(f)
        cont = pik.load(f)
        origfn = pik.load(f)
    return ells, mnts, cont, origfn


def load_waterline(waterlinedata):
    with open(waterlinedata, 'rb') as f:
        scale = pik.load(f)
        waterlevel= pik.load(f)
        newwaterlevel= pik.load(f)
        bright= pik.load(f)
    return scale, watelevel, newwaterlevel, brigt


def get_ell_data(ellsdict):
    """Returns list of frame no, x pos, y pos, and angle of ellipse"""
    frameno = []
    ellx = []
    elly = []
    ella = []
    ellw = []
    ellh = []
    for k in ellsdict.keys():
        frameno += [k]
        ((cx, cy), (w, h), ang) = ellsdict[k]
        ellx += [cx]
        elly += [cy]
        ellw += [w]
        ellh += [h]
        ella += [-1*(ang-90)]
    return frameno, ellx, elly, ella, ellw, ellh


def get_mnt_data(mntdict):
    frameno = []
    mntx= []
    mnty = []
    mnta = []
    for k in mntdict.keys():
        frameno += [k]
        (cx, cy) = mntdict[k]['center']
        ang = mntdict[k]['angle']
        mntx += [cx]
        mnty += [cy]
        mnta += [-1*ang]   
    return frameno, mntx, mnty, mnta

def get_mnt_data_outliers(cont, mntdict):
    tmp_frameno = []
    fin_frameno = []
    mntx= []
    mnty = []
    mnta = []
    areas = []
    for k in cont.keys():
        tmp_frameno += [k]
        areas += [cv2.contourArea(cont[k][0])]
    m_area = np.mean(areas)
    std_area = np.std(areas)
    for k in mntdict.keys():
        idx = tmp_frameno.index(k)
        if areas[idx] > m_area - std_area and areas[idx]< m_area + std_area:
            fin_frameno += [k]
            (cx, cy) = mntdict[k]['center']
            ang = mntdict[k]['angle']
            mntx += [cx]
            mnty += [cy]
            mnta += [-1*ang]   
    return fin_frameno, mntx, mnty, mnta



def view_watershed(crickfrogload):
    wat, mark, fn = cricketFrogLoad(crickfrogload)
    vid = vidHandlr(fn)
                
    cv2.namedWindow('orig', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
    cv2.namedWindow('wat', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
    cv2.namedWindow('ell', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )

    looping = True
    framelist = list(wat.keys())
    framelist.sort()
    
    curIdx = 0
    colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255
    
    while looping:
        curFr = framelist[curIdx]
        img = vid.get_frame(int(curFr), 'BGR')
        
        curWat = wat[curFr].toarray()
        
        overlay= colors[np.maximum(curWat*1, 0)]
        
        vis = cv2.addWeighted(img, 0.6, overlay, 0.4, 0.0, dtype=cv2.CV_8UC3)
        
        
        #fit ellipse
        mask = np.zeros(curWat.shape, dtype=np.uint8)
        mask[curWat] = 255
        
        contim, contpt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                             cv2.CHAIN_APPROX_NONE)
        #sort contours just in case
        cnts = sorted(contpt,  key=cv2.contourArea, reverse=True)[:5]   
        if len(cnts[0]) > 5:
            ell = cv2.fitEllipse(cnts[0])
            ((cx, cy), (_, _), ang) = ell
            
            ellim = img.copy()
            ellim = cv2.ellipse(ellim, ell, (0, 0, 255), 2)
            cv2.imshow('ell', ellim)
        else:
            cv2.imshow('ell', img)
        
        cv2.imshow('orig', img)
        cv2.imshow('wat', vis)
        
        k = cv2.waitKey(10)
        if k == 27: #escape
            cv2.destroyAllWindows()
            looping = False
            break
        elif k in [65363, ord('c'), ord('C')]: #right arrow
            if curIdx + 1 < len(framelist):
                curIdx += 1
        elif k in [65361, ord('z'), ord('Z')]: #left arrow
            if curIdx - 1 > 0:
                curIdx -= 1
        
def view_model_watershed(elload, modelload):
    
    mdl_ells, mdl_mnts, mdl_cont, origfn, mdl_segs = load_model_data(modelload)
    ells, mnts, cont, origfn = load_wat_ell(elload)
    
    print(origfn)
    vid = vidHandlr(origfn)
    print(vid.type)
    cv2.namedWindow('orig', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
    cv2.namedWindow('cont', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
    cv2.namedWindow('ell', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )

    looping = True
    framelist = list(range(vid.n_frames))
    framelist.sort()
    
    curIdx = 0
    #print(framelist)
    
    while looping:
        curFr = framelist[curIdx]
        img = vid.get_frame(int(curFr), 'BGR')
        cv2.imshow('orig', img)
        
        ## Contours
        
        cImg = img.copy()
        if curFr in cont.keys():
            cImg = cv2.drawContours(cImg, cont[curFr][0], -1, (255, 0, 0), 2)
        if curFr in mdl_cont.keys():
            cImg = cv2.drawContours(cImg, mdl_cont[curFr][0], -1, (0, 0, 255), 2)
            
        ## Ellipse
        
        eImg = img.copy()
        if curFr in ells.keys():
            eImg = cv2.ellipse(eImg, ells[curFr], (255, 0, 0), 2)
        if curFr in mdl_ells.keys():
            eImg = cv2.ellipse(eImg, mdl_ells[curFr], (0, 0, 255), 2)
            
        cv2.imshow('cont', cImg)
        cv2.imshow('ell', eImg)
        
        
        
        k = cv2.waitKey(10)
        if k == 27: #escape
            cv2.destroyAllWindows()
            looping = False
            break
        elif k in [65363, ord('c'), ord('C')]: #right arrow
            if curIdx + 1 < len(framelist):
                curIdx += 1
        elif k in [65361, ord('z'), ord('Z')]: #left arrow
            if curIdx - 1 > 0:
                curIdx -= 1
                
                
def butterworth_rms(data, fs=500):
    """Calculates RMS error of butterworth fit to data as a 
    function of frequency"""
    fny = fs / 2  # Nyquist frequency, fs = sampling frequency
    dt = 1 / fs  # measurement interval
    freq = np.arange(start= 1, stop=35, step=.1)
    rms_list = []
    for f in freq:
        #for each freq, fit butterworth
        N = 4 # order of filter
        Wn = f/fny # frequencies
        B, A = signal.butter(N, Wn, output='ba')
        #forwards backwards filter
        but_data = signal.filtfilt(B, A, np.array(data))
        data = np.array(data)
        #calculate rms
        rms = np.sum((but_data - np.array(data))**2)/len(data)
        
        rms_list += [rms]
    return freq, rms_list


def find_butterworth_cuttoff(data, fs=500):
    freq, rms_list = butterworth_rms(data, fs)
    
    for i in range(2, len(freq)):
        fnew = freq[::-1][0:i]
        rmsnew = rms_list[::-1][0:i]
    
        slope, intercept, r_value, p_value, std_err = linregress(fnew, rmsnew)
        if r_value**2 < 0.95:
            #print(fnew[-2])
            #print(intercept)
            break
            
        # optimize the cutoff frequency
        def tozero(freq_guess):
            return resid_interp(freq_guess) - intercept

        # interpolation function of the residuals
        resid_interp = interp1d(freq, rms_list, fill_value="extrapolate")
        fopt_guess = freq[np.argmin(np.abs(np.array(rms_list) - intercept))]
        #print('fopt_guess- %f' %fopt_guess)
        fopt = newton(tozero, fopt_guess, tol=10**(-4), maxiter=100)    
        #f_other = np.where(np.diff(np.sign(list(map(lambda x:(intercept-x), rms_list )))))[0][0]
        #print('mine- %f'  %freq[f_other])
        #print('isaac- %f' %fopt)
        return fopt


def but(data, freq, fs):
    #fs = sample frequency
    N = 2 # order of filter
    Wn = freq/(fs/2)
    B, A = signal.butter(N, Wn, output='ba')
    #forwards backwards filter
    fitdata = signal.filtfilt(B, A, np.array(data))
    return fitdata


def get_smooth_data(data, fs):
    #fs = sampling frequency
    a = find_butterworth_cuttoff(data, fs)
    smooth1 =  but(data, a, fs)
    smooth2 =  but(smooth1, a, fs)
    return smooth2


def findiff(p, dt):
    """Second-order accurate finite difference velocites and accelerations.

    Parameters:
    p = 1D array (size ntime) to take finite difference of
    dt = time step between measurements

    Returns:
    v = velocity
    a = acceleration

    Finite difference code from:
    See: https://en.wikipedia.org/wiki/Finite_difference_coefficient

    We are using 2nd order accurate central, forward, and backward finite
    differences.
    """

    n = len(p)
    v, a = np.zeros_like(p), np.zeros_like(p)

    # at the center
    for i in np.arange(1, n - 1):
        v[i] = .5 * p[i + 1] - .5 * p[i - 1]
        a[i] = p[i + 1] - 2 * p[i] + p[i - 1]

    # left boundary (forward difference)
    v[0] = -1.5 * p[0] + 2 * p[1] - .5 * p[2]
    a[0] = 2 * p[0] - 5 * p[1] + 4 * p[2] - p[3]

    # right boundary (backward differences)
    v[-1] = 1.5 * p[-1] - 2 * p[-2] + .5 * p[-3]
    a[-1] = 2 * p[-1] - 5 * p[-2] + 4 * p[-3] - p[-4]

    return v / dt, a / dt**2