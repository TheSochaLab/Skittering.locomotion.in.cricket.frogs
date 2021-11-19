import os, re, cv2
from imageio import imread
from Photron_Handlr import *

class vidHandlr():
    def __init__(self, filename):
        self.fn = filename
        
        if self.fn[-3:] == 'cih':
            self.info = cih_parser(self.fn)
            self.type = 'cih'
            self.vid = get_menmap(self.info)
            self.n_frames = self.vid.shape[0]
            self.height = self.vid.shape[1]
            self.width = self.vid.shape[2]
            self.cur_fr = 0
        
        #image stack from fastec
        elif os.path.isdir(self.fn):
            #set properties of self.info, self.vid = list
            self.info, self.vid = self.compile_list_file()
            self.type = 'imstack'
            
            if self.info is not None:
                self.n_frames = int(self.info['frame_count'])
                self.height = int(self.info['height'])
                self.width = int(self.info['width'])
            else:
                self.n_frames=len(self.vid)
                test = imread(self.fn + os.sep + self.vid[0]).shape
                self.height = test[0]
                self.width = test[1]
            self.cur_fr = 0    
        
        else:
            self.type = 'vid'
            self.vid = cv2.VideoCapture(self.fn)
            self.n_frames = int(self.vid.get(cv2.CAP_PROP_FRAME_COUNT))
            self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.info = None

            self.cur_fr = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
    def get_frame(self, frame, color='gray'):
        if self.type == 'vid':
            #set frame number
            self.vid.set(cv2.CAP_PROP_POS_FRAMES, frame)
            #get it
            ret, im = self.vid.read()
            self.cur_fr = self.vid.get(cv2.CAP_PROP_POS_FRAMES)
            if not ret:
                print("something is wrong")
            if color=='gray':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            elif color=='RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
        if self.type =='cih':
            #get frame
            im = self.vid[frame]
            
            cBits = int(self.info["EffectiveBit Depth"])
            nBits = int(self.info["Color Bit"])
            if self.info['EffectiveBit Side'] == "Lower":
                toshift = nBits - cBits
                im = np.left_shift(im, toshift)
            #cast to 8 bit    
            im = (im/256.0).astype('uint8')
            self.cur_fr = frame + 1
            if color=='BGR':
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            elif color=='RGB':
                im = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
            return im
        
        if self.type == 'imstack':
            #get frame
            imfn = self.vid[frame]
            self.cur_fr = frame + 1
            im = cv2.imread(self.fn + os.sep + imfn)
            if color=='gray':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            elif color=='RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            return im
            
            
        
    def compile_list_file(self):
        #metadata
        if os.path.exists(self.fn + os.sep + 'metadata.txt'):
            #then....
            with open(self.fn + os.sep + 'metadata.txt', 'r') as f:
                info = f.readlines()
            i2 = map(lambda k: k.split("="), info)
            metadata = [{x[0][1:]: x[1][:-1]} for x in list(i2) if len(x) == 2]
            metadata = { k: v for d in metadata for k, v in d.items() }
        else:
            metadata = None
        
        #get file list
        filelist = os.listdir(self.fn)
        imlist = [x for x in filelist if x[-3:].lower() == 'tif' and x[0] != '.']
        imlist.sort()
        
        return metadata, imlist