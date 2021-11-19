from vidHandlr import *
from dataHandlr import *
from waterline import *
from custom_watershed import *
import numpy as np



data_allfiles= {'AC01_04' : {'wat_ell':'DataAnalysis/AC01_04/2013-10-31 AC 01 04_000000-watershed-20180618-centang.pik',
                          'zoominfo':'DataAnalysis/AC01_04/2013-10-31 AC 01 04_000000-zoominfo-20180612.pik',
                          'fps':500,
                          'gait':'DataAnalysis/AC01_04/2013-10-31 AC 01 04_000000-gait-20180723.pik'
                         },
             'AC01_05' : {'wat_ell':'DataAnalysis/AC01_05/2013-10-31 AC 01 05_000000-watershed-20181003-centang_FIXED.pik',
                          'zoominfo':'DataAnalysis/AC01_05/2013-10-31 AC 01 05_000000-zoominfo-20180612.pik',
                          'fps':500,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC01_05/2013-10-31 AC 01 05_000000-gait-20180723.pik'
                         },
             'AC01_06' : {'wat_ell':'DataAnalysis/AC01_06/2013-10-31 AC 01 06_000000-watershed-20180621-centang.pik',
                          'zoominfo':'DataAnalysis/AC01_06/2013-10-31 AC 01 06_000000-zoominfo-20180614.pik',
                          'fps':500,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC01_06/2013-10-31 AC 01 06_000000-gait-20180723.pik'
                         },
             'AC01_10' : {'wat_ell':'DataAnalysis/AC01_10/2013-10-31 AC 01 10_000000-watershed-20180621-centang.pik',
                          'zoominfo':'DataAnalysis/AC01_10/2013-10-31 AC 01 10_000000-zoominfo-20180618.pik',
                          'fps': 250,
                          'gait': '/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC01_10/2013-10-31 AC 01 10_000000-gait-20180723.pik'
                         },
             ##
             ##
             'AC03_04' : {'wat_ell':'DataAnalysis/AC03_04/2013-10-31 AC 03 4_000000-watershed-20180622-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_04/2013-10-31 AC 03 4_000000-zoominfo-20180619.pik',
                          'fps': 250,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC03_04/2013-10-31 AC 03 4_000000-gait-20180723.pik'
                         },
             'AC03_06' :  {'wat_ell':'DataAnalysis/AC03_06/2013-10-31 AC 03 6_000000-watershed-20180620-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_06/2013-10-31 AC 03 6_000000-zoominfo-20180619.pik',
                          'fps': 250,
                           'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC03_06/2013-10-31 AC 03 6_000000-gait-20180723.pik'
                         
                         },
             'AC03_09' : {'wat_ell':'DataAnalysis/AC03_09/2013-10-31 AC 03 9_000000-watershed-20180620-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_09/2013-10-31 AC 03 9_000000-zoominfo-20180619.pik',
                          'fps': 250,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC03_09/2013-10-31 AC 03 9_000000-gait-20180725.pik'
                         },
             'AC03_14' : {'wat_ell':'DataAnalysis/AC03_14/2013-10-31 AC 03 14_000000-watershed-20180620-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_14/2013-10-31 AC 03 14_000000-zoominfo-20180619.pik',
                          'fps': 500,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC03_14/2013-10-31 AC 03 14_000000-gait-20180725.pik'
                         },
             'AC03_15' : {'wat_ell':'DataAnalysis/AC03_15/2013-10-31 AC 03 15_000000-watershed-20180620-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_15/2013-10-31 AC 03 15_000000-zoominfo-20180619.pik',
                          'fps': 500,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC03_15/2013-10-31 AC 03 15_000000-gait-20180723.pik'
                         },
             'AC03_18' : {'wat_ell':'DataAnalysis/AC03_18/2013-10-31 AC 03 18_000000-watershed-20180620-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_18/2013-10-31 AC 03 18_000000-zoominfo-20180619.pik',
                          'fps':500,
                          'gait':'/home/talcat/VTDrive/Talia Weiss/Projects/Cricket Frog Skittering/Analyses/CricketFrogPaper 052018/CricketSkittering/DataAnalysis/AC03_18/2013-10-31 AC 03 18_000000-gait-20180725.pik'
                         },
             'AC03_20': {'wat_ell':'DataAnalysis/AC03_20/2013-10-31 AC 03 20_000000-watershed-20180619-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_20/2013-10-31 AC 03 20_000000-zoominfo-20180619.pik',
                          'fps':500,
                         'gait':'/home/talcat/VTDrive/Talia Weiss/Projects/Cricket Frog Skittering/Analyses/CricketFrogPaper 052018/CricketSkittering/DataAnalysis/AC03_20/2013-10-31 AC 03 20_000000-gait-20180725.pik'
                         },
             'AC03_21' : {'wat_ell':'DataAnalysis/AC03_21/2013-10-31 AC 03 21_000000-watershed-20180611-centangEDIT.pik',
                          'zoominfo':'DataAnalysis/AC03_21/2013-10-31 AC 03 21_000000-zoominfo-20180524.pik',
                          'fps':500,
                          'gait':'/home/talcat/VTDrive/Talia Weiss/Projects/Cricket Frog Skittering/Analyses/CricketFrogPaper 052018/CricketSkittering/DataAnalysis/AC03_21/2013-10-31 AC 03 21_000000-gait-20180725.pik'
                         },
             'AC03_22' : {'wat_ell':'DataAnalysis/AC03_22/2013-10-31 AC 03 22_000000-watershed-20180622-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_22/2013-10-31 AC 03 22_000000-zoominfo-20180619.pik',
                          'fps':500,
                          'gait':'/home/talcat/VTDrive/Talia Weiss/Projects/Cricket Frog Skittering/Analyses/CricketFrogPaper 052018/CricketSkittering/DataAnalysis/AC03_22/2013-10-31 AC 03 22_000000-gait-20180725.pik'
                         },
             'AC03_23' :  {'wat_ell':'DataAnalysis/AC03_23/2013-10-31 AC 03 23_000000-watershed-20180619-centang.pik',
                          'zoominfo':'DataAnalysis/AC03_23/2013-10-31 AC 03 23_000000-zoominfo-20180619.pik',
                          'fps':500,
                          'gait':'/home/talcat/Desktop/Frog_Tools/CricketSkittering/DataAnalysis/AC03_23/2013-10-31 AC 03 23_000000-gait-20200330.pik'  #updated to fix error
                           #'gait':'/home/talcat/VTDrive/Talia Weiss/Projects/Cricket Frog Skittering/Analyses/CricketFrogPaper 052018/CricketSkittering/DataAnalysis/AC03_23/2013-10-31 AC 03 23_000000-gait-20180725.pik'
                         },
                
                ##
                ##
             'AC04_03' : {'wat_ell':'DataAnalysis/AC04_03/2013-10-31 AC 04 3_000000-watershed-20180619-centang.pik',
                          'zoominfo':'DataAnalysis/AC04_03/2013-10-31 AC 04 3_000000-zoominfo-20180619.pik',
                          'fps':250,
                          'gait':'/home/talcat/VTDrive/Talia Weiss/Projects/Cricket Frog Skittering/Analyses/CricketFrogPaper 052018/CricketSkittering/DataAnalysis/AC04_03/2013-10-31 AC 04 3_000000-gait-20180726.pik'
                         }
            
            }


maxVelIdx = {'AC01_04':[151],
             'AC01_05':[17,164],
             'AC01_06':[107,299],
             'AC01_10':[133],
             
             'AC03_04':[20,102],
             'AC03_06':[80],
             'AC03_09':[18],
             'AC03_14':[225],
             'AC03_15':[59,242],
             'AC03_18':[35,194],
             'AC03_20':[39,212],
             'AC03_21':[183,454],
             'AC03_22':[226,501],
             'AC03_23':[37,201],
             
             'AC04_03':[46, 132],
            }

frog_size = {'AC01_04':2.01,
             'AC01_05':2.01,
             'AC01_06':2.01,
             'AC01_10':2.01,
             
             'AC03_04':2.14,
             'AC03_06':2.14,
             'AC03_09':2.14,
             'AC03_14':2.14,
             'AC03_15':2.14,
             'AC03_18':2.14,
             'AC03_20':2.14,
             'AC03_21':2.14,
             'AC03_22':2.14,
             'AC03_23':2.14,
             
             'AC04_03':2.23,
            }
          

allfrog_rev = {
'AC03_22' : False,
'AC03_20' : False,
'AC03_04' : True,
'AC01_05' : False,
'AC01_10' : False,
'AC03_15' : True, 
'AC03_06': False,
'AC01_06': False,
'AC03_14': True,
'AC01_04': False,
'AC03_21': False,
'AC03_23': False,
'AC03_18': True,
'AC03_09': True,
'AC04_03': False    
}

            
def loadSingleFrogwVideo(frog_seq):
    if frog_seq in data_allfiles.keys():
        key = frog_seq
        ells, mnts, cont, origfn = load_wat_ell(data_allfiles[key]['wat_ell'])
        with open(data_allfiles[key]['zoominfo'], 'rb') as f:
            scale = pik.load(f)
            waterlevel= pik.load(f)
            newwaterlevel= pik.load(f)
            bright= pik.load(f)

        vid = vidHandlr(origfn)
        h = vid.height
        w = vid.width    

        frameno, mntx, mnty, mnta = get_mnt_data(mnts)
        frameno, ellx, elly, ella, ellw, ellh = get_ell_data(ells)
        watercontactIdx = np.where(np.diff(np.sign(list(map(lambda x:(h-x) - (h - newwaterlevel), mnty )))))

        if 'gait' in data_allfiles[key].keys() and data_allfiles[key]['gait'] is not None:
            with open(data_allfiles[key]['gait'], 'rb') as f:
                gait = pik.load(f)
            HL_t = [i + 1 for i, (x, y) in enumerate(zip(gait['HL'], gait['HL'][1:])) if x != y and (x == 't' or y == 't')]
            HL_a = [i + 1 for i, (x, y) in enumerate(zip(gait['HL'], gait['HL'][1:])) if x != y and (x == 'a' or y == 'a')]
            FL_t = [i + 1 for i, (x, y) in enumerate(zip(gait['FL'], gait['FL'][1:])) if x != y and (x == 't' or y == 't')]
            FL_a = [i + 1 for i, (x, y) in enumerate(zip(gait['FL'], gait['FL'][1:])) if x != y and (x == 'a' or y == 'a')]
            
            FL_t = [(y,(FL_t[x +1] - y ) )for x, y in enumerate(FL_t) if x%2 == 0]
            HL_t = [(y,(HL_t[x +1] - y ) )for x, y in enumerate(HL_t) if x%2 == 0]
            HL_a = [(y,(HL_a[x +1] - y) ) for x, y in enumerate(HL_a) if x%2 == 0]   
            FL_a = [(y,(FL_a[x +1] - y) ) for x, y in enumerate(FL_a) if x%2 == 0]   
        else:
            HL_t = None
            HL_a = None
            FL_t = None
            FL_a = None
        data = { 'fn': origfn,
                      'frameno': np.array(frameno),
                      'time': np.array(frameno)/data_allfiles[key]['fps'],
                      'height': h,
                      'width': w,
                      'fps': data_allfiles[key]['fps'],
                      'waterlevel': newwaterlevel,
                      'watercontactIdx': watercontactIdx,
                      'mntx': np.array(mntx),
                      'mnty': newwaterlevel - np.array(mnty),
                      'o_mnty':np.array(mnty),
                      'mnta': np.array(mnta),
                      'ellx': np.array(ellx),
                      'elly': newwaterlevel - np.array(elly),
                      'ella': np.array(ella),
                      'ellh': np.array(ellh),
                      'ellw': np.array(ellw),
                      'pxSVL': np.mean(ellh[10:-10]),
                
                      'HL_t':HL_t,
                      'HL_a':HL_a,
                      'FL_t':FL_t,
                      'FL_a':FL_a
                    }
        return data
    else:
        print('Not a valid sequence')
        return None

    
def loadSingleFrogwVideo_smoothed_watercontactIdx(frog_seq):
    if frog_seq in data_allfiles.keys():
        key = frog_seq
        ells, mnts, cont, origfn = load_wat_ell(data_allfiles[key]['wat_ell'])
        with open(data_allfiles[key]['zoominfo'], 'rb') as f:
            scale = pik.load(f)
            waterlevel= pik.load(f)
            newwaterlevel= pik.load(f)
            bright= pik.load(f)

        vid = vidHandlr(origfn)
        h = vid.height
        w = vid.width    

        frameno, mntx, mnty, mnta = get_mnt_data(mnts)
        frameno, ellx, elly, ella, ellw, ellh = get_ell_data(ells)
        s_mnty = get_smooth_data(mnty, data_allfiles[key]['fps'])
        watercontactIdx = np.where(np.diff(np.sign(list(map(lambda x:(h-x) - (h - newwaterlevel), s_mnty )))))

        if 'gait' in data_allfiles[key].keys() and data_allfiles[key]['gait'] is not None:
            with open(data_allfiles[key]['gait'], 'rb') as f:
                gait = pik.load(f)
            HL_t = [i + 1 for i, (x, y) in enumerate(zip(gait['HL'], gait['HL'][1:])) if x != y and (x == 't' or y == 't')]
            HL_a = [i + 1 for i, (x, y) in enumerate(zip(gait['HL'], gait['HL'][1:])) if x != y and (x == 'a' or y == 'a')]
            FL_t = [i + 1 for i, (x, y) in enumerate(zip(gait['FL'], gait['FL'][1:])) if x != y and (x == 't' or y == 't')]
            FL_a = [i + 1 for i, (x, y) in enumerate(zip(gait['FL'], gait['FL'][1:])) if x != y and (x == 'a' or y == 'a')]
            
            FL_t = [(y,(FL_t[x +1] - y ) )for x, y in enumerate(FL_t) if x%2 == 0]
            HL_t = [(y,(HL_t[x +1] - y ) )for x, y in enumerate(HL_t) if x%2 == 0]
            HL_a = [(y,(HL_a[x +1] - y) ) for x, y in enumerate(HL_a) if x%2 == 0]   
            FL_a = [(y,(FL_a[x +1] - y) ) for x, y in enumerate(FL_a) if x%2 == 0]   
        else:
            HL_t = None
            HL_a = None
            FL_t = None
            FL_a = None
        data = { 'fn': origfn,
                      'frameno': np.array(frameno),
                      'time': np.array(frameno)/data_allfiles[key]['fps'],
                      'height': h,
                      'width': w,
                      'fps': data_allfiles[key]['fps'],
                      'waterlevel': newwaterlevel,
                      'watercontactIdx': watercontactIdx,
                      'mntx': np.array(mntx),
                      'mnty': newwaterlevel - np.array(mnty),
                      'o_mnty':np.array(mnty),
                      'mnta': np.array(mnta),
                      'ellx': np.array(ellx),
                      'elly': newwaterlevel - np.array(elly),
                      'ella': np.array(ella),
                      'ellh': np.array(ellh),
                      'ellw': np.array(ellw),
                      'pxSVL': np.mean(ellh[10:-10]),
                
                      'HL_t':HL_t,
                      'HL_a':HL_a,
                      'FL_t':FL_t,
                      'FL_a':FL_a
                    }
        return data
    else:
        print('Not a valid sequence')
        return None    
    
    
    
def resave_paper_friendly(frogseq):
    data = loadSingleFrogwVideo(frogseq)
    #fix reversing for plot purposes
    if allfrog_rev[frogseq]:
        data['mnta'] = -1*data['mnta']
        data['mntx'] = (np.max(data['mntx']) - data['mntx'])
        
    else:
        data['mntx'] = data['mntx'] - np.min(data['mntx'])

    #save the smoothed data
    data['s_mntx'] = get_smooth_data(data['mntx'], data['fps'])
    data['s_mnty'] =get_smooth_data(data['mnty'], data['fps'])
    data['s_mnta'] = get_smooth_data(data['mnta'], data['fps'])
    data['frogsize(cm)'] =frog_size[frogseq] 
    
    with open('DataAnalysis/PaperFriendly/%s.pik'%frogseq, 'wb') as f:
        pik.dump(data, f)
        
        
def resave_paper_friendly2(frogseq):
    data = loadSingleFrogwVideo_smoothed_watercontactIdx(frogseq)
    #fix reversing for plot purposes
    if allfrog_rev[frogseq]:
        data['mnta'] = -1*data['mnta']
        data['mntx'] = (np.max(data['mntx']) - data['mntx'])
        
    else:
        data['mntx'] = data['mntx'] - np.min(data['mntx'])

    #save the smoothed data
    data['s_mntx'] = get_smooth_data(data['mntx'], data['fps'])
    data['s_mnty'] =get_smooth_data(data['mnty'], data['fps'])
    data['s_mnta'] = get_smooth_data(data['mnta'], data['fps'])
    data['frogsize(cm)'] =frog_size[frogseq] 
    
    with open('DataAnalysis/PaperFriendly2/%s.pik'%frogseq, 'wb') as f:
        pik.dump(data, f)

