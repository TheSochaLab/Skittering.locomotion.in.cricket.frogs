import numpy as np
import pickle as pik
import matplotlib.pyplot as plt
#import plotly
#import plotly.plotly as ply
#import plotly.graph_objs as go

import matplotlib
params = {
    'text.latex.preamble': ['\\usepackage{gensymb}'],
    'image.origin': 'lower',
    'image.interpolation': 'nearest',
    'image.cmap': 'gray',
    'axes.grid': False,
    'savefig.dpi': 150,  # to adjust notebook inline plot size
    'axes.labelsize': 14, # fontsize for x and y labels (was 10)
    'axes.titlesize': 20,
    'font.size': 14, # was 10
    'legend.fontsize': 10, # was 10
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'text.usetex': False,
    'figure.figsize': [3.39, 2.10],
    'font.family': 'serif',
     'legend.fontsize': 12

}
matplotlib.rcParams.update(params)


def loadSingleFrog(frogseq):
    """Given the sequences digitized, will load the friendly versions of the files based on their relative locations
    in this script.  These can be changed here if you are examining the data in the supplementary material"""
    listOfFiles = {
            'AC01_04': 'DataAnalysis/PaperFriendly/AC01_04.pik',
            'AC01_05': 'DataAnalysis/PaperFriendly/AC01_05.pik',
            'AC01_06': 'DataAnalysis/PaperFriendly/AC01_06.pik',
            'AC01_10': 'DataAnalysis/PaperFriendly/AC01_10.pik',
             
            'AC03_04': 'DataAnalysis/PaperFriendly/AC03_04.pik',
            'AC03_06': 'DataAnalysis/PaperFriendly/AC03_06.pik',
            'AC03_09': 'DataAnalysis/PaperFriendly/AC03_09.pik',
            'AC03_14': 'DataAnalysis/PaperFriendly/AC03_14.pik',
            'AC03_15': 'DataAnalysis/PaperFriendly/AC03_15.pik',
            'AC03_18': 'DataAnalysis/PaperFriendly/AC03_18.pik',
            'AC03_20': 'DataAnalysis/PaperFriendly/AC03_20.pik',
            'AC03_21': 'DataAnalysis/PaperFriendly/AC03_21.pik',
            'AC03_22': 'DataAnalysis/PaperFriendly/AC03_22.pik',
            'AC03_23': 'DataAnalysis/PaperFriendly/AC03_23.pik',
             
            'AC04_03': 'DataAnalysis/PaperFriendly/AC04_03.pik',
    }
    if frogseq not in listOfFiles.keys():
        print('Not a valid sequence')
        return None
    else:
        with open(listOfFiles[frogseq], 'rb') as f:
            data = pik.load(f)
        return data
    
def loadSingleFrog_wsmoothed(frogseq):
    """Given the sequences digitized, will load the friendly versions of the files based on their relative locations
    in this script.  These can be changed here if you are examining the data in the supplementary material"""
    listOfFiles = {
            'AC01_04': 'DataAnalysis/PaperFriendly/AC01_04.pik',
            'AC01_05': 'DataAnalysis/PaperFriendly/AC01_05.pik',
            'AC01_06': 'DataAnalysis/PaperFriendly/AC01_06.pik',
            'AC01_10': 'DataAnalysis/PaperFriendly/AC01_10.pik',
             
            'AC03_04': 'DataAnalysis/PaperFriendly/AC03_04.pik',
            'AC03_06': 'DataAnalysis/PaperFriendly/AC03_06.pik',
            'AC03_09': 'DataAnalysis/PaperFriendly/AC03_09.pik',
            'AC03_14': 'DataAnalysis/PaperFriendly/AC03_14.pik',
            'AC03_15': 'DataAnalysis/PaperFriendly/AC03_15.pik',
            'AC03_18': 'DataAnalysis/PaperFriendly/AC03_18.pik',
            'AC03_20': 'DataAnalysis/PaperFriendly/AC03_20.pik',
            'AC03_21': 'DataAnalysis/PaperFriendly/AC03_21.pik',
            'AC03_22': 'DataAnalysis/PaperFriendly/AC03_22.pik',
            'AC03_23': 'DataAnalysis/PaperFriendly/AC03_23.pik',
             
            'AC04_03': 'DataAnalysis/PaperFriendly/AC04_03.pik',
    }
    if frogseq not in listOfFiles.keys():
        print('Not a valid sequence')
        return None
    else:
        with open(listOfFiles[frogseq], 'rb') as f:
            data = pik.load(f)
        return data    


def graph_all(key, time='ms', figsize=(10, 16), segment=False, clip=False):
    data = loadSingleFrog_wsmoothed(key)
    scale = data['frogsize(cm)']
   

    fig, ax = plt.subplots(nrows = 4, sharex = True, figsize = figsize)
    


    ### unc = np.std(data['ellh'])/data['pxSVL'] #uncertainty in SVL size
    
    if time == 'ms':
        thetime = data['time']*1000
        ftt = 1000/data['fps']
    if time == 'sec':
        thetime = data['time']
        ftt = 1/data['fps']
    if time == 'frame':
        thetime = data['frameno']
        ftt = 1
    
    
       
    ###########
    # Strict jump definition
    ###########
    if segment:
        fr_thestart = data['HL_a'][0][0]
        fr_theend = data['HL_a'][1][0]
        indices = [i for i, j in enumerate(data['frameno']) if j == fr_thestart]
        if len(indices) == 0:
            thestart = 0
            print("Need to fix beginning data")
        else:    
            thestart = indices[0]
        indices = [i for i, j in enumerate(data['frameno']) if j == fr_theend]
        theend = indices[0]
        thetime2 = thetime[thestart:theend+1]
        #thetime = thetime - thetime[0]
        for axs in ax:
            axs.set_xlim(thetime2[0], thetime2[-1])
    else:
        thestart = 0
        theend = len(thetime)
       
    

    
    ############
    # GAIT
    ############
    
    if data['HL_a'] is not None:
        HL_a = [(x*ftt, y*ftt) for (x, y) in data['HL_a']]
        HL_t = [(x*ftt, y*ftt) for (x, y) in data['HL_t']]
        FL_t = [(x*ftt, y*ftt) for (x, y) in data['FL_t']]
        FL_a = [(x*ftt, y*ftt) for (x, y) in data['FL_a']]
        
        for t_val in HL_t:
            for a_val in HL_a:
                if np.abs(a_val[0] - (t_val[0] + t_val[1])) < 25*ftt:
                    #print(t_val, a_val)
                    start = (t_val[0] + t_val[1])
                    end = a_val[0] - start
                    hatch_bar = ax[3].broken_barh([(start, end)], (1, 2), facecolors='gray', hatch='////')
        
        
        black_bar = ax[3].broken_barh(HL_a, (1, 2), facecolors='black')
        gray_bar = ax[3].broken_barh(HL_t, (1, 2), facecolors='gray')
        ax[3].broken_barh(FL_a, (3.5, 2), facecolors='black')
        ax[3].broken_barh(FL_t, (3.5, 2), facecolors='gray')
        for val in data['HL_a']:
            cx = val[0] + val[1]/2
            cy = 1 + 2/2
            cx = cx*ftt
    
            #ax[3].annotate('A', (cx, cy), color='w', weight='bold', fontsize=12, ha='center', va='center')

        for val in data['HL_t']:
            cx = val[0] + val[1]/2
            cy = 1 + 2/2
            cx = cx*ftt
    
            #ax[3].annotate('T', (cx, cy), color='w', weight='bold', fontsize=12, ha='center', va='center')
    
        for val in data['FL_t']:
            cx = val[0] + val[1]/2
            cy = 3.5 + 2/2
            cx = cx*ftt
    
            #ax[3].annotate('T', (cx, cy), color='w', weight='bold', fontsize=12, ha='center', va='center')
    
        for val in data['FL_a']:
            cx = val[0] + val[1]/2
            cy = 3.5 + 2/2
            cx = cx*ftt
    
            #ax[3].annotate('A', (cx, cy), color='w', weight='bold', fontsize=12, ha='center', va='center')
            
        ax[3].set_yticks([2, 4.5])
        ax[3].set_yticklabels(['HL', 'FL'])
        ax[3].set_ylabel('Gait', rotation=0, va='center', ha='right')
        ax[3].legend([black_bar, gray_bar, hatch_bar], ['Away', 'Towards', 'Transition'])
     
    
    
    ########
    #  X Position
    ########
    mntx = data['mntx']  #[thestart:theend]
    s_mntx = data['s_mntx'] #[thestart:theend]
    
    ax[0].plot(thetime, mntx/data['pxSVL']*scale, 'k.', label='raw data', zorder=2)
    #ax[0].scatter(thetime, mntx/data['pxSVL'], c=thetime, label='raw data', cmap='gnuplot')
    
    
    #get uncertainty using delta method
    ###v_smntx = np.gradient(s_mntx/data['pxSVL'], data['time'])
    ###test = (unc)**2*v_smntx**2/len(data['ellh'])
    ###test =  np.sqrt(test)
    
    ##ax[0].fill_between(thetime, s_mntx/data['pxSVL'] - test, s_mntx/data['pxSVL'] + test, facecolor='gray', interpolate ='True', zorder =1)


    ax[0].plot(thetime, s_mntx/data['pxSVL']*scale, 'r-', label='smoothed data', zorder=3)
    

    ax[0].set_ylabel('Horizontal\nposition\n(cm)', rotation=0, ha='right', va='center')

    ax[0].legend()

    axt = ax[0].twinx()
    mn, mx = ax[0].get_ylim()
    axt.set_ylim(mn/scale, mx/scale)
    axt.set_ylabel('(SVL)', rotation=0, ha='left', va='center')

    #######################
    #  Y POSITION
    #######################
    mnty = data['mnty'] #[thestart:theend]
    s_mnty = data['s_mnty'] #[thestart:theend]
    
    ax[1].plot(thetime, mnty/data['pxSVL']*scale, 'k.', label='orig', zorder=2)


    ax[1].plot(thetime, s_mnty/data['pxSVL']*scale, 'r-', label='smooth', zorder=3)
    
    #get uncertainty using delta method
    ###v_smnty = np.gradient(s_mnty/data['pxSVL'], data['time'])
    ###test = (unc)**2*v_smnty**2/len(data['ellh'])
    ###test =  np.sqrt(test)
    
    
    ##ax[1].fill_between(thetime,  data['s_mnty']/data['pxSVL'] - test, data['s_mnty']/data['pxSVL'] + test, facecolor='gray', interpolate ='True', zorder=1)


    ax[1].set_ylabel('Vertical\nposition\n(cm)', rotation=0, va='center', ha='right')
    ax[1].axhline(0, color='c', label='water line')

    #plt.legend()

    axt = ax[1].twinx()
    mn, mx = ax[1].get_ylim()
    axt.set_ylim(mn/scale, mx/scale)
    axt.set_ylabel('(SVL)', rotation=0, ha='left', va='center')

    #############################
    # BODY ANGLE
    ###########################

    mnta = data['mnta'] #[thestart:theend]
    
    ax[2].plot(thetime, mnta, 'k.', label='orig')

    s_mnta = data['s_mnta'] #[thestart:theend]  
                 
    ax[2].plot(thetime, s_mnta, 'r-', label='smooth')
    
    en = int(len(thetime)/30)
    
    ax[2].quiver(thetime[::en],np.ones(thetime.shape)[::en] *(np.min(data['mnta'] - 15)), np.cos(np.deg2rad(s_mnta))[::en],np.sin(np.deg2rad(s_mnta))[::en],
         headwidth=1, headlength=3)
    
    
    
    
    
    ax[2].set_ylabel('Body\nAngle\n(deg)', rotation=0, va='center', ha='right')

    ax[3].set_xlabel('Time (%s)' %time)
    
    
    
    #######################
    # Underwater bars
    #######################
    
    
    
    water = [0] + list(data['watercontactIdx'][0]) + [len(data['frameno']) - 1]
    #print(s_mnty[water])
    test = [(water[i], water[i+1]) for i in range(len(water) - 1) if s_mnty[int(np.mean([water[i], water[i+1]]))] < 0]
    
    for idx, tax in enumerate(ax):             
        for o in test:
            if idx == 0:
                y_max = 1
                y_min = 0
            else:
                y_min = 0
                y_max = 1.2
            if o[0] == 0:
                start = np.array(thetime)[o[0]] - 10*ftt
            else:
                start = np.array(thetime)[o[0]] 
            if o[1] == len(s_mnty) - 1:
                end = np.array(thetime)[o[1]] + 10 *ftt
            else:
                end = np.array(thetime)[o[1]]
            if clip:
                if start < thetime2[0]:
                    start = thetime2[0]
                if end > thetime2[-1]:
                    end = thetime2[-1]
                
               
            tax.axvspan(start, end, ymin = y_min, ymax = y_max, facecolor='cadetblue', alpha = 1, zorder = -1, clip_on=False)
            
    
    ax[0].set_title(key)
                 
        

    
    if segment:
        for idx, tax in enumerate(ax):             
            for o in range(2):
                if idx == 0:
                    y_max = 1
                    y_min = 0
                else:
                    y_min = 0
                    y_max = 1.2
                if o == 0:
                    start = np.array(thetime)[0] - 10*ftt
                    end = np.array(thetime2[0])
                if o == 1:
                    start = np.array(thetime2[-1])
                    end = np.array(thetime)[-1] + 10 *ftt
                    
                if clip:
                    for axs in ax:
                        axs.set_xlim(thetime2[0], thetime2[-1])
                    
                else:            
                    tax.axvspan(start, end, ymin = y_min, ymax = y_max, facecolor='white', alpha = .7, zorder = 5, clip_on=False)
        
        
        #for axs in ax:
        #    axs.set_xlim(thetime2[0], thetime2[-1])
    
        #set bounds
        
    if clip:
            
        ax[3].set_xbound([thetime2[0], thetime2[-1]])
    else:
        xmin = np.array(thetime)[water[0]] - 10*ftt
        xmax = np.array(thetime)[water[-1]] + 10*ftt
        ax[3].set_xbound([xmin, xmax])
   

    return ax, fig



def graph_vel(key, time='ms', figsize = (8, 8), segment=False, clip=False):
    data = loadSingleFrog_wsmoothed(key)
    scale = data['frogsize(cm)']


    fig, ax = plt.subplots(nrows = 3, sharex = True, figsize =figsize)


    
    if time == 'ms':
        thetime = data['time']*1000
        ftt = 1000/data['fps']
    if time == 'sec':
        thetime = data['time']
        ftt = 1/data['fps']
    if time == 'frame':
        thetime = data['frameno']
        ftt = 1

    
        
       
    ###########
    # Strict jump definition
    ###########
    if segment:
        fr_thestart = data['HL_a'][0][0]
        fr_theend = data['HL_a'][1][0]
        indices = [i for i, j in enumerate(data['frameno']) if j == fr_thestart]
        if len(indices) == 0:
            thestart = 0
            print("Need to fix beginning data")
        else:    
            thestart = indices[0]
        indices = [i for i, j in enumerate(data['frameno']) if j == fr_theend]
        theend = indices[0]
        thetime2 = thetime[thestart:theend+1]
        #thetime = thetime - thetime[0]
        for axs in ax:
            axs.set_xlim(thetime2[0], thetime2[-1])
    else:
        thestart = 0
        theend = len(thetime)
       
    
    ###POSITION INFO

    mntx = data['mntx']
    s_mntx = data['s_mntx']/data['pxSVL']*scale
    s_mnty = data['s_mnty']/data['pxSVL']*scale
    
    ####
    # Horizontal Velocity
    ####
    v_smntx=np.gradient(s_mntx, data['time'])
    ax[0].plot(thetime, v_smntx, 'k.', label='orig')

    ax[0].set_ylabel('Horizontal\nvelocity\n(cm/sec)',rotation=0, ha='right', va='center')

    

    axt = ax[0].twinx()
    mn, mx = ax[0].get_ylim()
    axt.set_ylim(mn/scale, mx/scale)
    axt.set_ylabel('(SVL/sec)', rotation=0, ha='left', va='center')
    
    
    ####
    # Vertical Velocity
    ####
    v_smnty= np.gradient(s_mnty, data['time'])
    
    ax[1].plot(thetime, v_smnty, 'k.', label='orig')

    ax[1].set_ylabel('Vertical\nvelocity\n(cm/sec)',rotation=0, ha='right', va='center')

    axt = ax[1].twinx()
    mn, mx = ax[1].get_ylim()
    axt.set_ylim(mn/scale, mx/scale)
    axt.set_ylabel('(SVL/sec)', rotation=0, ha='left', va='center')

    ####
    # Total Velocity
    ####
    
    v_tot = np.sqrt(v_smnty**2 + v_smntx**2)
    
    ax[2].plot(thetime, v_tot, 'k.', label='orig')

    ax[2].set_ylabel('Total\nvelocity\n(cm/sec)',rotation=0, ha='right', va='center')

    axt = ax[2].twinx()
    mn, mx = ax[2].get_ylim()
    axt.set_ylim(mn/scale, mx/scale)
    axt.set_ylabel('(SVL/sec)', rotation=0, ha='left', va='center')

    
    water = [0] + list(data['watercontactIdx'][0]) + [len(data['frameno']) - 1]
    #print(s_mnty[water])
    test = [(water[i], water[i+1]) for i in range(len(water) - 1) if s_mnty[int(np.mean([water[i], water[i+1]]))] < 0]
    
    for idx, tax in enumerate(ax):             
        for o in test:
            if idx == 0:
                y_max = 1
                y_min = 0
            else:
                y_min = 0
                y_max = 1.2
            if o[0] == 0:
                start = np.array(thetime)[o[0]] - 10*ftt
            else:
                start = np.array(thetime)[o[0]] 
            if o[1] == len(s_mnty) - 1:
                end = np.array(thetime)[o[1]] + 10 *ftt
            else:
                end = np.array(thetime)[o[1]]
            tax.axvspan(start, end, ymin = y_min, ymax = y_max, facecolor='cadetblue', alpha = .5, zorder = 0, clip_on=False)
            
    
    ax[0].set_title(key)
    ax[2].set_xlabel('Time (%s)' %time)
                 
        
    #set bounds
    xmin = np.array(thetime)[water[0]] - 10*ftt
    xmax = np.array(thetime)[water[-1]] + 10*ftt
    ax[2].set_xbound([xmin, xmax])
    
    for axs in ax:
        axs.axhline(0, c='r')
    
       
    if segment:
        for idx, tax in enumerate(ax):             
            for o in range(2):
                if idx == 0:
                    y_max = 1
                    y_min = 0
                else:
                    y_min = 0
                    y_max = 1.2
                if o == 0:
                    start = np.array(thetime)[0] - 10*ftt
                    end = np.array(thetime2[0])
                if o == 1:
                    start = np.array(thetime2[-1])
                    end = np.array(thetime)[-1] + 10 *ftt
                    
                if clip:
                    for axs in ax:
                        axs.set_xlim(thetime2[0], thetime2[-1])
                    
                else:            
                    tax.axvspan(start, end, ymin = y_min, ymax = y_max, facecolor='white', alpha = .7, zorder = 5, clip_on=False)
    
    
    if clip:
            
        ax[2].set_xbound([thetime2[0], thetime2[-1]])
    else:
        xmin = np.array(thetime)[water[0]] - 10*ftt
        xmax = np.array(thetime)[water[-1]] + 10*ftt
        ax[2].set_xbound([xmin, xmax])
    
    
    #remove xaxis
    #for tax in ax:
        #print(idx)
    #    for loc, spine in tax.spines.items():
    #        if loc in ['bottom', 'top']:
    #            spine.set_color('none') # don't draw spine

    
    #ob = AnchoredHScaleBar(size=100, label="100 ms", loc=4, frameon=False,
    #                   pad=0.6,sep=4,color="k")
    #ax[3].add_artist(ob)
    #ax[3].axvline(data['time'][151]*1000, color='c', label='Max vertical velocity')
    return ax, fig
       
    #adjust_spines(ax[2], ['left', 'right'])
    #adjust_spines(ax[3], ['left', 'right', 'bottom'])
    

    
    
    
    
    
    
def graph_xy2(key, time='ms',figsize = (8, 4)):
    data = loadSingleFrog2(key)
    scale = data['frogsize(cm)']
    

    fig, ax = plt.subplots(nrows = 1, sharex = True, figsize = figsize)

    unc = np.std(data['ellh'])/data['pxSVL'] #uncertainty in SVL size
    
    if time == 'ms':
        thetime = data['time']*1000
        ftt = 1000/data['fps']
    if time == 'sec':
        thetime = data['time']
        ftt = 1/data['fps']
    if time == 'frame':
        thetime = data['frameno']
        ftt = 1

   
    ####
    # Graphing x v  y
    ####
    s_mntx = data['s_mntx']
    s_mnty = data['s_mnty']
    
    #get uncertainty using delta method
    v_smnty = np.gradient(s_mnty/data['pxSVL']*scale, data['time'])
    test = (unc)**2*v_smnty**2/len(data['ellh'])
    test =  np.sqrt(test)
    
    ax.fill_between(s_mntx/data['pxSVL']*scale, s_mnty/data['pxSVL']*scale - test, s_mnty/data['pxSVL']*scale + test, facecolor='gray', interpolate ='True', zorder =1)
    sc = ax.scatter((s_mntx/data['pxSVL']*scale)[::2], (s_mnty/data['pxSVL']*scale)[::2], c=thetime[::2], cmap='gnuplot', marker='.')
    cbar = plt.colorbar(sc, ax=ax)
    cbar.ax.set_title('Time (%s)'%time)
    #ax.plot(s_mntx/data['pxSVL'], s_mnty/data['pxSVL'], 'r-', label='smooth')
    

    ax.set_xlabel('Horizontal Position (cm)')
    ax.set_ylabel('Vertical\nposition\n(cm)', rotation=0, ha='right', va='center')
    ax.axhline(0, color='c', label='water line')

    #plt.legend()
    return ax, fig
