from vidHandlr import *
from Photron_Handlr import *
import cv2, os, time
import numpy as np
from scipy.sparse import csr_matrix
import pickle as pik

class App:
    """This takes a video filename, tries to load it into a video Object with vidHandlr, and allows 
    one to perform watershed on each frame with a gamma control on the bottom
    c -> forward a frame
    z -> backward a frame
    1, 2, 3, 4, 5, 6, 7 -> marker colors.  Only 1 and 2 are supported with the rest of this code
    r -> reset the drawn markers"""
    def __init__(self, video_fn, frame_int=1):
        #video object
        self.fn = video_fn
        self.vid = vidHandlr(video_fn)
        self.cur_frame = 0
        #how many frames to skip 
        self.frame_int = frame_int
        #Just markers of current frame
        self.markers = np.zeros((self.vid.height, self.vid.width), dtype=np.uint8)
        #markers for entire video
        self.vid_markers = [csr_matrix((self.vid.height, self.vid.width), dtype=np.uint8)]*self.vid.n_frames
        
        #watershed dictionary of video
        self.watershed = {}
        
        self.cur_marker = 1 #1 - 7
        
        #list of possible marker colors
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255
        
        #This is true with left button down
        self.drawing = False 
        #self.dragging = False
        #self.zooming = False
        
        #for drawing
        self.prevPt = None
        
        #brightness, contrast
        self.alpha = 1
    
        

                
        
        
        
        #windows
        self.drawingWin = 'drawing'
        self.watershedWin = 'watershed'
        cv2.namedWindow(self.drawingWin,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
        
        cv2.namedWindow(self.watershedWin,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
        
        cv2.setMouseCallback(self.drawingWin, self.on_mouse)
        
        cv2.imshow(self.drawingWin, self.getAdjustIm())
        cv2.imshow(self.watershedWin, self.getAdjustIm())
        
        cv2.displayOverlay(self.drawingWin, "Frame %d/%d    Marker %d" %(self.cur_frame, self.vid.n_frames, self.cur_marker), 0)
    
    
        #brightness, contrast?
        cv2.createTrackbar("gamma", self.drawingWin, 0, 255, lambda x:x )
        cv2.setTrackbarMin('gamma', self.drawingWin, 0)
    
    def getAdjustIm(self):
             
        im = self.vid.get_frame(self.cur_frame, 'BGR')
        
        #get HSV
        hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
        
        hsv = hsv.astype(np.float64)
        
        hsv[:,:,2]= hsv[:,:,2] * (10*self.alpha/(255) + 1.0)
        
        themax = np.max(hsv)
        #print(themax)
        
        hsv[hsv > 255] = 255
        
        hsv = hsv.astype('uint8')
        
        im2 = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return im2.astype(np.uint8)
        
    
    def get_color(self, marker):
        """gets the current color"""
        return list(map(int, self.colors[marker]))
    
    def on_mouse(self, event, x, y, flags, param):
        pt = (x, y)
        #if event == cv2.EVENT_LBUTTONDOWN:
        #    
        #    if flags & cv2.EVENT_FLAG_SHIFTKEY:
        #        self.dragging= True
        if event == cv2.EVENT_RBUTTONDOWN:
            self.drawing=True
            self.prevPt = pt
        #elif event== cv2.EVENT_LBUTTONUP:
        #    self.dragging = False
            
        elif event== cv2.EVENT_RBUTTONUP:
            self.drawing = False
            self.prevPt = None
            
        if self.drawing and flags & cv2.EVENT_FLAG_RBUTTON:
            #Get markers matrix for current frame
            self.markers = self.vid_markers[self.cur_frame].toarray()
            #self.markers = self.markers.astype('int32')
            #Draw current markers in position
            if self.prevPt:
                temp = np.zeros(self.markers.shape)
                temp= cv2.line(temp, self.prevPt, (x, y), 255, 5)
                self.prevPt = (x, y)
                temp = temp == 255
                self.markers[temp] = self.cur_marker
            
                #write this frames markers to vid_markers
                self.vid_markers[self.cur_frame] = csr_matrix(self.markers)
            
                #call function to update windows due to drawing
                self.drawWindowUpdate()
            

    
    def drawWindowUpdate(self):
        """Actually draws on frame and shows drawn on image"""
        dirty_im = self.getAdjustIm()
        self.markers = self.vid_markers[self.cur_frame].toarray()
        for m in range(1, 8): #go through markers 1- 7
            #get all markers of that color
            mask = self.markers==m
            #get color
            color = self.get_color(m)
            
            #write colors to dirty im
            dirty_im[mask, :] = color
        
            #show dirtyim
            cv2.imshow(self.drawingWin, dirty_im)
        cv2.displayOverlay(self.drawingWin, "Frame %d/%d    Marker %d" %(self.cur_frame, self.vid.n_frames, self.cur_marker), 0)
            
    def drawWatershedUpdate(self):
        """To be run after drawWindowUpdate, will actually watershed"""
        dirty_im = self.getAdjustIm()
        self.markers = self.vid_markers[self.cur_frame].toarray()
        m = self.markers.copy().astype('int32')
        if np.count_nonzero(m) != 0 and not self.drawing:  
            water = cv2.watershed(dirty_im, m)
            overlay = self.colors[np.maximum(water, 0)]
            vis = cv2.addWeighted(dirty_im, 0.6, overlay, 0.4, 0.0, dtype=cv2.CV_8UC3)
            cv2.imshow(self.watershedWin, vis)
        
            self.watershed[self.cur_frame] = csr_matrix(water)

            
    def run(self):
        while True:
            k= cv2.waitKeyEx(5) 
            if k == 27: #Escape
                cv2.destroyAllWindows()
                break
            #Set marker color
            elif k>= ord('1')and k <= ord('7'): 
                self.cur_marker = k - ord('0')
                print('marker: ', self.cur_marker)
                cv2.displayOverlay(self.drawingWin, "Frame %d/%d    Marker %d" %(self.cur_frame, self.vid.n_frames, self.cur_marker), 0)
            #clean all marks off of image
            elif k in [ord('r'), ord('R')]:
                #make new clean image
                self.markers = np.zeros((self.vid.height, self.vid.width), dtype=np.int32)
                #clear the frame in refence 
                self.vid_markers[self.cur_frame] = csr_matrix(self.markers)
                
                #update
                #self.drawWindowUpdate()
            
            #go through frames
            elif k in [65363, ord('c'), ord('C')]: #right arrow
                new_frame = self.cur_frame + self.frame_int
                if new_frame <= self.vid.n_frames:
                    #print("frame: ", self.cur_frame)
                    self.cur_frame = new_frame
                    self.cur_marker = 1
                    #self.drawWindowUpdate()
                    self.drawWatershedUpdate()
                #else:
                #    cv2.displayOverlay(self.drawingWin, "Cannot go further", 20)
                    
            elif k in [65361, ord('z'), ord('Z')]: #left arrow
                new_frame = self.cur_frame - self.frame_int
                if new_frame >= 0:
                    #print("frame: ", self.cur_frame)
                    self.cur_frame = new_frame
                    self.cur_marker = 1
                    #self.drawWindowUpdate()
                    #self.drawWatershedUpdate()
                #else:
                #    cv2.displayOverlay(self.drawingWin, "Cannot go further", 20)
            
            #drawing the watershed
            elif k == 65509: #CAPSLOCK
                self.drawWatershedUpdate()
            elif k != -1:
                print("Key: ", k)
                
            self.alpha = cv2.getTrackbarPos('gamma', self.drawingWin)
            self.drawWindowUpdate()
            self.drawWatershedUpdate()
            

def cricketFrogSave(app):
    """Given the watershed app with marker values, will save the raw marker positions, the watershed
    segmentation, and the filename of the video"""
    #filename = os.path.basename(a.fn)
    fn_list = app.fn.split('/')
    fn_list = [x for x in fn_list if x != 'wb' and x != 'scaled']
    filename = fn_list[-1]
    
    date = time.strftime('%Y%m%d')
    
    filenamefull = 'DataAnalysis/%s-watershed-%s.pik' %(filename,date)
    
    marks = app.vid_markers
    watershed = app.watershed
    
    #change to boolean sparse for space
    for key in watershed.keys():
        watershed[key] = watershed[key] == 1
    
    
    with open(filenamefull, 'wb') as f:
        pik.dump(watershed, f)
        pik.dump(marks, f)
        pik.dump(app.fn, f)
        

def cricketFrogLoad(filenamefull):
    """Will load the raw marker positions, the watershed segmentation, and video filename from a pickle
    file saved with cricketFrogSave"""
    with open(filenamefull, 'rb') as f:
        watershed = pik.load(f)
        marks = pik.load(f)
        filename = pik.load(f)
    return watershed, marks, filename


def findEllipse(crickwatshed):
    """Loads the watershed segmentation and marks file, then fits a contour to the segmentation
    will calculate the moments of the contour to get the angle and center position of the contour, as
    well as fitting an ellipse to the contour (giving a center position, height, width, and angle).
    Will then save all contour and ellipse information in a pickle file of the same input name"""
    watershed, marks, origfn = cricketFrogLoad(crickwatshed)
    filename = crickwatshed[:-4] + '-centang.pik'
    
    ellcent = {}
    mntcent = {}
    contours = {}
    for k in watershed.keys():
        print(k)
        seg = watershed[k].toarray()
        mask = np.zeros(seg.shape, dtype=np.uint8)
        mask[seg] = 255
        
        contim, contpt, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        print(len(contpt))
        #in case mutlipe contours
        cnts = sorted(contpt,  key=cv2.contourArea, reverse=True)[:5] 
        #print(len(cnts[0]))
        contours[k] = cnts
        #ellipse center and angle
        
        ell = cv2.fitEllipse(cnts[0])
        ellcent[k] = ell
        #contour center and angle
        mnt = cv2.moments(cnts[0])
        cx = mnt['m10']/mnt['m00']
        cy = mnt['m01']/mnt['m00']
        ang = 1/2*np.arctan2(2*mnt['mu11']/mnt['m00'], (mnt['mu20']/mnt['m00'] - mnt['mu02']/mnt['m00']))
        
        mntcent[k] = {'center': (cx, cy), 'angle':np.rad2deg(ang)}
        
    with open(filename, 'wb') as f:
        pik.dump(ellcent, f)
        pik.dump(mntcent, f)
        pik.dump(contours, f)
        pik.dump(origfn, f)
        
        
def showContourEllipse(imdir, conts):
    """Given the pickle file with the contours (from findEllipse), will draw the contours 
    and ellipses on the video to show what it looks like."""
    vid = vidHandlr(imdir)
    with open(conts, 'rb') as f:
        ells = pik.load(f)
        mnts = pik.load(f)
        cont = pik.load(f)
            
    cv2.namedWindow('cont', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
    cv2.namedWindow('ellipse', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
        
    framelist = list(ells.keys())
    framelist.sort()
    curIdx = 0
        
    running = True
    while running:
        thekey = framelist[curIdx]
            
        image = vid.get_frame(int(thekey), 'BGR')
        imCnt = image.copy()
        imEll = image.copy()
            
        imCnt = cv2.drawContours(imCnt, cont[thekey], -1, (255, 0, 0), 2)
        cntang = -1*mnts[thekey]['angle']
            
        imEll = cv2.ellipse(imEll, ells[thekey], (0, 0, 255), 2)
        ((cx, cy), (_, _), ellang) = ells[thekey]
            
        cv2.imshow('cont', imCnt)
        cv2.imshow('ellipse', imEll)
        cv2.displayOverlay('cont', "Angle %f" %(cntang), 0)
        cv2.displayOverlay('ellipse', "Angle %f" %(-1*(ellang-90)), 0)
            
        k = cv2.waitKey(10)
        if k == 27: #Escape
            cv2.destroyAllWindows()
            running=False
            break
        elif k in [65363, ord('c'), ord('C')]: #right arrow
            if curIdx + 1 < len(framelist):
                curIdx += 1
        elif k in [65361, ord('z'), ord('Z')]: #left arrow
            if curIdx - 1 > 0:
                curIdx -= 1
                
        
