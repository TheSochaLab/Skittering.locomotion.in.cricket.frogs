from vidHandlr import *
from Photron_Handlr import *
import cv2, os, time
import numpy as np
from scipy.sparse import csr_matrix
import pickle as pik

class App:
    def __init__(self, video_fn, frame_int=1):
        #video object
        self.fn = video_fn
        self.vid = vidHandlr(video_fn)
        self.cur_frame = 0
        #how many frames to skip 
        self.frame_int = frame_int
                
        #This is true with left button down
        self.drawing = False 
        #self.dragging = False
        #self.zooming = False
        
        self.scale_sep = 1000
        
        #for drawing
        self.line = 0
        self.zoomline = 0 
        #self.overlay = np.zeros((self.vid.height, self.vid.width), dtype=np.bool)
        
        #scaling top portion
        self.scale = 0
        self.trans = 0
        
        #brightness, contrast
        self.alpha = 1
    
        #list of possible marker colors
        self.colors = np.int32( list(np.ndindex(2, 2, 2)) ) * 255
        
        #windows
        self.drawingWin = 'drawing'
        self.zoomWin = 'zoom'
        
        cv2.namedWindow(self.drawingWin,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
        cv2.namedWindow(self.zoomWin,cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL )
        
        
        cv2.setMouseCallback(self.drawingWin, self.on_mouse)
        
        cv2.imshow(self.drawingWin, self.getAdjustIm())
        cv2.imshow(self.zoomWin, self.getTransformIm())
        
        cv2.displayOverlay(self.drawingWin, "Frame %d/%d   Scale %3f" %(self.cur_frame, self.vid.n_frames, self.scale/self.scale_sep + 1), 0)
    
    
        #brightness, contrast?
        cv2.createTrackbar("gamma", self.drawingWin, 0, 255, lambda x:x )
        cv2.setTrackbarMin('gamma', self.drawingWin, 0)
    
        #scale above image
        cv2.createTrackbar("ScaleTop", self.drawingWin, 0, self.scale_sep, lambda x:x )
        cv2.setTrackbarMin('ScaleTop', self.drawingWin, 0)
    
    
        self.run()
    
    
    def scaleTranslate(self):
        scale = self.scale/self.scale_sep + 1
        cur_frame = self.getAdjustIm()
        
        if self.line != 0: #we have a line defined
            cur_frame = self.getAdjustIm()
            y = self.line
            topIm = cur_frame[:y, :, :]
            botIm = cur_frame[y:, :, :]
            
            #rescale top
            scTop = cv2.resize(topIm, (0, 0), fx = scale, fy = scale)
            
            th, tw, _ = scTop.shape
            self.zoomline = th
            bh, bw, _ = botIm.shape
            
            #new image size
            newImage = np.zeros((th+bh, tw, 3), np.uint8)
            #set the scaled top 
            newImage[:th, :, :] = scTop
            #center the scaled bottom
            center = int(tw/2)
            left = center - bw/2
            right = center + bw/2
            newImage[th:, int(left):int(right), :] = botIm
            
            return newImage
        else:
            return cur_frame
    
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
        
    
    def getTransformIm(self):
        return self.scaleTranslate()
    
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
            self.line = y
        #elif event== cv2.EVENT_LBUTTONUP:
        #    self.dragging = False
            
        elif event== cv2.EVENT_RBUTTONUP:
            self.drawing = False
            
            
        if self.drawing and flags & cv2.EVENT_FLAG_RBUTTON:

            #Draw line
            #dirty_im = self.getTransformIm()
            #h, w, _ = dirty_im.shape

            #temp = np.zeros((h, w))
            #temp= cv2.line(temp, (0, y), (w, y), (255), 1)
            
            #self.overlay = temp==255
            #self.line = y
            
            #call function to update windows due to drawing
            self.drawWindowUpdate()
            
            
    
    def drawWindowUpdate(self):
        """Actually draws on frame and shows drawn on image"""
        dirty_im = self.getAdjustIm()
        zoom = self.getTransformIm()
        #print(self.overlay.shape)
        #print(dirty_im.shape)
        #dirty_im[self.overlay] = (0, 0, 255)
        h, w, _ = dirty_im.shape
        hz, wz, _ = zoom.shape
        y = self.line
        yz = self.zoomline
        temp = cv2.line(dirty_im, (0, y), (w, y), (0, 0, 255), 3)
        temp2 = cv2.line(zoom, (0, yz), (wz, yz), (0, 0, 255), 3)
        
        cv2.imshow(self.drawingWin, dirty_im)
        cv2.imshow(self.zoomWin, zoom)
        cv2.displayOverlay(self.drawingWin, "Frame %d/%d  Scale %3f" %(self.cur_frame, self.vid.n_frames, self.scale/self.scale_sep+ 1), 0)
            

            
    def run(self):
        while True:
            k= cv2.waitKeyEx(5) 
            if k == 27 : #Escape
                cv2.destroyAllWindows()
                y = self.line
                print("Waterline is at y position %d" %y)
                print("Zoom is at level %f" %(self.scale/self.scale_sep + 1))
                #self.saveZoomLevel()
                break
 
            
            
            #go through frames
            elif k in [65363, ord('c'), ord('C')]: #right arrow
                new_frame = self.cur_frame + self.frame_int
                if new_frame < self.vid.n_frames:
                    #print("frame: ", self.cur_frame)
                    self.cur_frame = new_frame
                    self.cur_marker = 1
                    #self.drawWindowUpdate()

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

            elif k != -1:
                print("Key: ", k)
                
            self.alpha = cv2.getTrackbarPos('gamma', self.drawingWin)
            self.scale = cv2.getTrackbarPos('ScaleTop', self.drawingWin)
            
            
            self.drawWindowUpdate()

    def saveZoomLevel(self):
        """Given the watershed app with marker values, will save things"""
        #filename = os.path.basename(a.fn)
        fn_list = self.fn.split('/')
        filename = fn_list[-1]
        if filename == 'wb':
            filename = fn_list[-2]
        date = time.strftime('%Y%m%d')
    
        fullfilename = 'DataAnalysis/%s-zoominfo-%s.pik' %(filename, date)
    
        scale = self.scale
        waterlevel = self.line
        new_waterlevel = self.zoomline
        bright = self.alpha
  
        with open(fullfilename, 'wb') as f:
            pik.dump(scale, f)
            pik.dump(waterlevel, f)
            pik.dump(new_waterlevel, f)
            pik.dump(bright, f)
            
    def resave_images(self):
        """will make a new directory and resave the new scaled images to disk"""
        newdir = self.fn + '/scaled'
        os.mkdir(newdir)
        
        for fr in range(self.vid.n_frames):
            self.cur_frame = fr
            im = self.getTransformIm()
            cv2.imwrite(newdir + '/%03d.tif'%(fr), im)
            