import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk, Image, ImageGrab
from tkinter import filedialog
from scipy import signal, spatial, ndimage
import time
import datetime
import os
import random
from CustomFunctions import *
import threading

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)
tk.Canvas.create_circle = _create_circle

def openfn():
        filename = filedialog.askopenfilename(title='open')
        return filename

class App:

    def open_img(self,panel):
        x = openfn()
        img = Image.open(x)
        self.change_img(panel,img)

    def change_img(self,panel,image):
        img = ImageTk.PhotoImage(image)
        #print(isinstance(panel, tk.Canvas))
        if isinstance(panel, tk.Canvas):
            panel.itemconfigure(1,image = img)
            panel.image = img
            panel.config(width=img.width(), height=img.height())
        else:    
            panel.configure(image=img)
            panel.image = img
    
    def open_original_img(self):
        self.stop.set()
        self.cap.release()
        #global X
        x = openfn()
        if not x: return
        img = Image.open(x)
        self.X = np.array(img.convert('L'),dtype='single');
        self.change_img(self.window.nametowidget("leftframe.originalimage"),img)
        #window.nametowidget("leftframe.originalimage").config(width=X.shape, height=200)

    def open_template_img(self):
        #global kernel
        x = openfn()
        if not x: return
        img = Image.open(x)
        self.Template = img
        self.kernel = np.array(img.convert('L'),dtype='single');
        self.change_img(self.window.nametowidget("rightframe.templateimage"),img)

    def camera_feed(self):
        print("changed to camera feed")
        self.video_thread = threading.Thread(target=self.video_playback)
        self.video_thread.start()

        #window.after(33, camera_feed)

    def change_method(self,val):
        print("change method to " + str(val))
        return 0

    def Viola_Jones(self,X:np.array):
        frame_gray = cv2.equalizeHist(np.array(X,dtype='uint8'))
        #-- Detect faces
        faces = self.face_cascade.detectMultiScale(frame_gray)
        faces = list(faces)
        faces.sort(key=lambda a: -a[2]*a[3])
        eyes = []
        #print(faces)
        for (x,y,w,h) in faces[:1]:
            center = (x + w//2, y + h//2)
            #frame = cv2.ellipse(frame, center, (w//2, h//2), 0, 0, 360, (255, 0, 255), 4) ##########    

            faceROI = frame_gray[y:y+h,x:x+w]
            #-- In each face, detect eyes
            eyes = self.eyes_cascade.detectMultiScale(faceROI)
            eyes = list(eyes)
            eyes.sort(key=lambda x: -x[2]*x[3])

            #for (x2,y2,w2,h2) in eyes[:2]:
            #    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
            #    radius = int(round((w2 + h2)*0.25))
            #    #frame = cv.circle(frame, eye_center, radius, (255, 0, 0 ), 4) #############3
 
        return faces[:1],eyes[:2]

        #cv2.imshow('Capture - Face detection', frame) ##############

    def apply_algorithm(self):
        meth = self.method_selection_variable.get()
        ind = self.OPTIONS.index(meth)
        #print('applying method '+meth+' under number ',ind)
        #method = eval(meth)    
        #global pos
        if (ind>=11):
            tic = time.perf_counter()
            self.faces,self.eyes = self.Viola_Jones(self.X)
            self.eyes.sort(key=lambda a: a[0])
            toc = time.perf_counter()
            if self.faces:
                (x,y,w,h) = self.faces[0]
                self.window.nametowidget("leftframe.originalimage").coords(3,x,y,x+w,y+h)
            else: 
                self.window.nametowidget("leftframe.originalimage").coords(3,0,0,0,0)
            if self.eyes:
                (x2,y2,w2,h2) = self.eyes[0]
                self.window.nametowidget("leftframe.originalimage").coords(4,x+x2,y+y2,x+x2+w2,y+y2+h2)
                if len(self.eyes)>=2:
                    (x2,y2,w2,h2) = self.eyes[1]
                    self.window.nametowidget("leftframe.originalimage").coords(5,x+x2,y+y2,x+x2+w2,y+y2+h2)
                else: self.window.nametowidget("leftframe.originalimage").coords(5,0,0,0,0)
            else:
                self.window.nametowidget("leftframe.originalimage").coords(4,0,0,0,0)
                self.window.nametowidget("leftframe.originalimage").coords(5,0,0,0,0)

           
            self.window.nametowidget("leftframe.originalimage").coords(2,0,0,0,0)
            self.window.nametowidget("rightframe.infobox").configure(text="time: " + "{:.6f}".format(toc-tic)+"\n"+"Viola-Jones")
            
            if (ind==12): self.find_symmetry_lines()
            else:
                    self.window.nametowidget("leftframe.originalimage").coords(6,0,0,0,0)
                    self.window.nametowidget("leftframe.originalimage").coords(7,0,0,0,0)
                    self.window.nametowidget("leftframe.originalimage").coords(8,0,0,0,0)
            return
    
        method = eval(meth)    
        if (ind<5): 
            #print("custom func")
            tic = time.perf_counter()
            res, self.pos = method(self.X,self.kernel)
            toc = time.perf_counter()
            self.pos = [self.pos[1],self.pos[0]]
        if (ind>=5): 
            #print("cv2 library func")
            tic = time.perf_counter()
            res = cv2.matchTemplate(self.X,self.kernel,method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            toc = time.perf_counter()
            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
            else:
                top_left = max_loc
            bottom_right = (top_left[0] + self.kernel.shape[0], top_left[1] + self.kernel.shape[1])
            self.pos = top_left

        self.faces = [[self.pos[0],self.pos[1],self.kernel.shape[1],self.kernel.shape[0]]]
        self.eyes = []

        self.change_img(self.window.nametowidget("rightframe.resultimage"),Image.fromarray(normalize(res)))
        self.window.nametowidget("rightframe.infobox").configure(text="time: " + "{:.6f}".format(toc-tic)+"\n"+meth)
        self.window.nametowidget("rightframe.resultimage").coords(2,self.pos[0]-1,self.pos[1]-1,self.pos[0]+1,self.pos[1]+1)
        self.window.nametowidget("leftframe.originalimage").coords(2,self.pos[0],self.pos[1],self.pos[0]+self.kernel.shape[1],self.pos[1]+self.kernel.shape[0])

        self.window.nametowidget("leftframe.originalimage").coords(3,0,0,0,0)
        self.window.nametowidget("leftframe.originalimage").coords(4,0,0,0,0)
        self.window.nametowidget("leftframe.originalimage").coords(5,0,0,0,0)
        self.window.nametowidget("leftframe.originalimage").coords(6,0,0,0,0)
        self.window.nametowidget("leftframe.originalimage").coords(7,0,0,0,0)
        self.window.nametowidget("leftframe.originalimage").coords(8,0,0,0,0)

    def find_symmetry_lines(self):
        eyeline = [-1,-1,-1,-1]
        faceline = [-1,-1,-1,-1]
        eyesim=[]
        facecrop = []
        if self.faces:
            (x,y,w,h) = self.faces[0]
            center = (x + w//2, y + h//2)
            if len(self.eyes)>=2:
                #we have two eyes data
                for i in range(2):
                    (x2,y2,w2,h2) = self.eyes[i]
                    eye_center = (x + x2 + w2//2, y + y2 + h2//2)
                    eyeline[i*2:i*2+2] = [eye_center[0],eye_center[1]]
                v = np.array([eyeline[2]-eyeline[0],eyeline[3]-eyeline[1]])
                v=v/np.linalg.norm(v)
                v2 = np.array([-v[1],v[0]])
                angle = np.arcsin(v[1])
                #rotation = spatial.transform.Rotation.from_euler('z', angle, degrees=False)
                facecrop = ndimage.rotate(self.X[y:y+h,x:x+w],np.degrees(angle),reshape=False)
                shift = findSymmetry(facecrop)
                shift = shift - w//2
                center1 = [center[0]+v[0]*shift,center[1]+v[1]*shift]
                faceline[0:2] = [center1[0]-v2[0]*h//2,center1[1]-v2[1]*h//2]
                faceline[2:4] = [center1[0]+v2[0]*h//2,center1[1]+v2[1]*h//2]
                #cv2.imshow('Capture - Face detection', facecrop/256) ##############
            else:
                facecrop = self.X[y:y+h,x:x+w]
                shift = findSymmetry(facecrop)
                shift = shift - w//2
                center1 = [center[0]+shift,center[1]]
                faceline[0:2] = [center1[0],center1[1]-h//2]
                faceline[2:4] = [center1[0],center1[1]+h//2]
                #we don't have eyes data
                pass

            self.window.nametowidget("leftframe.originalimage").coords(6,faceline[0],faceline[1],faceline[2],faceline[3])
        else: 
            self.window.nametowidget("leftframe.originalimage").coords(6,0,0,0,0)
        if self.faces and self.eyes:
            #(x2,y2,w2,h2) = self.eyes[0]
            #self.window.nametowidget("leftframe.originalimage").coords(7,x+x2,y+y2,x+x2+w2,y+y2+h2)
            if len(self.eyes)>=2:
                #v,v2,angle are avalible
                for i in range(2):
                    (x2,y2,w2,h2) = self.eyes[i]
                    center = (x + x2 + w2//2, y + y2 + h2//2)
                    facecrop = ndimage.rotate(self.X[y+y2:y+y2+h2,x+x2:x+x2+w2],np.degrees(angle),reshape=False)
                    shift = findSymmetry(facecrop)
                    shift = shift - w2//2
                    center1 = [center[0]+v[0]*shift,center[1]+v[1]*shift]
                    tmp = [center1[0]-v2[0]*h2//2,center1[1]-v2[1]*h2//2,center1[0]+v2[0]*h2//2,center1[1]+v2[1]*h2//2]
                    #eyesim=eyesim+[tmp]
                    #(x2,y2,w2,h2) = self.eyes[1]
                    self.window.nametowidget("leftframe.originalimage").coords(7+i,tmp[0],tmp[1],tmp[2],tmp[3])
            else:
                (x2,y2,w2,h2) = self.eyes[0]
                center = (x + x2 + w2//2, y + y2 + h2//2)
                facecrop = self.X[y+y2:y+y2+h2,x+x2:x+x2+w2]
                shift = findSymmetry(facecrop)
                shift = shift - w2//2
                center1 = [center[0]+shift,center[1]]
                tmp = [center1[0],center1[1]-h2//2,center1[0],center1[1]+h2//2]
                #eyesim=eyesim+[tmp]
                #(x2,y2,w2,h2) = self.eyes[1]
                self.window.nametowidget("leftframe.originalimage").coords(7,tmp[0],tmp[1],tmp[2],tmp[3])
                self.window.nametowidget("leftframe.originalimage").coords(8,0,0,0,0)
        else:
            self.window.nametowidget("leftframe.originalimage").coords(7,0,0,0,0)
            self.window.nametowidget("leftframe.originalimage").coords(8,0,0,0,0)
        return

    def video_playback(self):
        self.stop.clear()
        print("changed to camera feed")
        #global X
        self.cap.open(0)
        while not self.stop.is_set():
            tic = time.perf_counter()
            ret, frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not ret:
                print("error in reading frame")
                return
            img = Image.fromarray(frame)
            self.X = np.array(img.convert('L'),dtype='single');
            self.change_img(self.window.nametowidget("leftframe.originalimage"),img)
            toc = time.perf_counter()
            self.apply_algorithm()
            if (toc-tic<0.016): time.sleep(0.016-(toc-tic))

    def save_png(self):
        # save postscipt image
        canvas = self.window.nametowidget("leftframe.originalimage")
        t = datetime.datetime.now()
        filename = "{}.png".format(t.strftime("%Y-%m-%d_%H-%M-%S"))
        filename = 'screenshots/'+ filename
        self.window.update() 
        x=self.window.winfo_rootx()#+canvas.winfo_x()
        y=self.window.winfo_rooty()#+canvas.winfo_y()
        x1=x+self.window.winfo_width()#canvas.winfo_width()
        y1=y+self.window.winfo_height()#canvas.winfo_height()
        ImageGrab.grab().crop((x,y,x1,y1)).save(filename) # works only when windows system screen scaling option is at 100%
        #img = ImageGrab.grab()
        #img.save(filename)
    
    def __init__(self):

        #os.chdir("C:\\Users\\xxxxx\\source\\repos\\Task2\\Task2") 
        self.OPTIONS = ['TemplateMatchingBasic','CrossCorrelationBasic','TemplateMatchingFourier','CrossCorrelationFourier','CrossCorrelationMeanCorrectedFourier',
                   'cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR','cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED','Viola_Jones','Viola_Jones & face symmetry']

        self.Original = Image.open('image.png')
        self.Template = Image.open('kernel.png')
        self.X = np.array(self.Original.convert('L'),dtype='single')#/256.0
        self.kernel = np.array(self.Template.convert('L'),dtype='single')#/256.0
        #kernel = np.ones([80,70],dtype='single')

        self.window = tk.Tk()
        self.window.title("Task 2")

        self.frame_a = tk.Frame(master=self.window,name = "leftframe" )
        self.frame_b = tk.Frame(master=self.window,name = "rightframe" )
        self.frame_a.pack(side='left')
        self.frame_b.pack(side='right')

        self.image1 = ImageTk.PhotoImage(self.Original)
        #original_image.image = image1

        tic = time.perf_counter()
        img_2, tmp = CrossCorrelationFourier(self.X,self.kernel)
        toc = time.perf_counter()
        self.image2 = ImageTk.PhotoImage(Image.fromarray(normalize(img_2)))
        #result_image.image = image2

        self.image3 = ImageTk.PhotoImage(self.Template)

        self.method_selection_variable = tk.StringVar(self.window)
        self.method_selection_variable.set(self.OPTIONS[0]) # default value
        #w = tk.OptionMenu(window, variable, *OPTIONS).pack()

 
        label_a = tk.Label(master=self.frame_a, text="original image:",height=1,width=20).pack()#side='top')
        #original_image = tk.Label(master=frame_a, text="image",bg='red',image=image1,name = "originalimage").pack()#side='top')
        original_image = tk.Canvas(master=self.frame_a,name = "originalimage")
        original_image.pack()#side='top')
        original_image_canvas = original_image.create_image(0, 0,anchor=tk.NW, image=self.image1)

        label_b = tk.Label(master=self.frame_b, text="Result",height=1,width=20).pack()#side='top')
        #result_image = tk.Label(master=frame_b, text="Result",bg='green',image=image2,name = "resultimage").pack()#side='top')
        result_image = tk.Canvas(master=self.frame_b,name = "resultimage")
        result_image.pack()#side='top')
        result_image_canvas = result_image.create_image(0, 0,anchor=tk.NW, image=self.image2)

        label_c = tk.Label(master=self.frame_a, text="Select method:",height=1,width=20).pack()#side='top')
        method_selection = tk.OptionMenu(self.frame_a, self.method_selection_variable, *self.OPTIONS, command=self.change_method).pack()

        label_scale = tk.Label(master=self.frame_a, text="template_scaling:",height=1,width=20).pack()#side='top')
        self.e_s = tk.Entry(master=self.frame_a, width=6)
        self.e_s.pack()
        btn1 = tk.Button(master=self.frame_a, text='scale template', command=self.scale_template).pack()
        btn1 = tk.Button(master=self.frame_a, text='auto mode', command=self.auto_mode).pack()
        
        
        label_d = tk.Label(master=self.frame_b, text="Template:",height=1,width=20).pack()#side='top')
        template_image = tk.Label(master=self.frame_b, text="info panel",image=self.image3,name = "templateimage").pack()#side='top')
        label_e = tk.Label(master=self.frame_b, text="Info:",height=1,width=20).pack()#side='top')
        info = tk.Label(master=self.frame_b, text="time: ",bg='grey',name = "infobox").pack()#side='top')

        btn1 = tk.Button(master=self.frame_b, text='open original image', command=self.open_original_img).pack()
        btn2 = tk.Button(master=self.frame_b, text='open template image', command=self.open_template_img).pack()
        btn3 = tk.Button(master=self.frame_b, text='change to camera feed', command=self.camera_feed).pack()
        btn4 = tk.Button(master=self.frame_b, text='Analyze', command=self.apply_algorithm).pack()
        btn5 = tk.Button(master=self.frame_b, text='Take snapshot', command=self.save_png).pack()
        #command=lambda: action(someNumber)

        #canvas = tk.Canvas(master=frame_a,bg='red')
        #canvas.pack(fill='both', expand=1)
        #canvas.create_image(0, 0, anchor=tk.NW, image=image1)

        self.pos = [0,0]
        original_image.create_rectangle(self.pos[0], self.pos[1], self.pos[0], self.pos[1],
                           fill=None,
                           outline='red',
                           width=3)
        result_image.create_circle(self.pos[0], self.pos[1], 2,
                           fill='red',
                           outline='red',
                           width=3)

        original_image.create_oval(0,0,0,0,fill=None,outline='red',width=3)
        original_image.create_oval(0,0,0,0,fill=None,outline='blue',width=3)
        original_image.create_oval(0,0,0,0,fill=None,outline='blue',width=3)

        original_image.create_line(0,0,0,0,fill='green',width=2)
        original_image.create_line(0,0,0,0,fill='green',width=2)
        original_image.create_line(0,0,0,0,fill='green',width=2)

        width, height = 1280, 720
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        #ret, frame = cap.read()

        self.face_cascade = cv2.CascadeClassifier()
        self.eyes_cascade = cv2.CascadeClassifier()
        face_cascade_name = 'data/haarcascade_frontalface_default.xml'
        eyes_cascade_name = 'data/haarcascade_eye.xml'
        if not self.face_cascade.load(cv2.samples.findFile(face_cascade_name)):
            print('--(!)Error loading face cascade')
            exit(0)
        if not self.eyes_cascade.load(cv2.samples.findFile(eyes_cascade_name)):
            print('--(!)Error loading eyes cascade')
            exit(0)
        self.faces = []
        self.eyes = []

        self.window.bind('<Escape>', quit)
        self.window.protocol("WM_DELETE_WINDOW", self.quit)
        #threading.Thread.__init__(self)

        self.stop = threading.Event()
        self.video_thread = None#threading.Thread(target=self.video_playback)
        #self.video_thread.start()

        #cap.release()

    def scale_template(self):
        scale = None
        try:
            scale = float(self.e_s.get())
        except:
            pass
        if not scale or scale<=0.01:# or scale>=1:
            scale = 1.0
        self.e_s.delete(0, "end")
        self.e_s.insert(0, scale)
        newscale = self.Template.size
        newscale = (int(scale*newscale[0]),int(scale*newscale[1]))

        if scale<=1:
            img = cv2.resize(np.array(self.Template), newscale, interpolation = cv2.INTER_AREA)
        else:
            img = cv2.resize(np.array(self.Template), newscale, interpolation = cv2.INTER_CUBIC)
        if len(img.shape)==3:
            self.kernel = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY),dtype='single');
        if len(img.shape)==1:
            self.kernel = np.array(img,dtype='single');
        self.change_img(self.window.nametowidget("rightframe.templateimage"),Image.fromarray(img))


    def run(self):
        self.window.mainloop()

    def quit(self):
        self.stop.set()
        time.sleep(0.1)
        self.cap.release()
        time.sleep(0.1)
        self.window.destroy()
        #self.window.quit()

    def close_window(self):
        self.running = False  # turn off while lo

    def auto_mode(self):
        window1 = tk.Toplevel(self.window)
        window1.protocol("WM_DELETE_WINDOW", self.close_window)
        self.running = True
        data = read_data_from_disk()
        label1 = tk.Label(master = window1, text="TemplateMatchingFourier",height=1,width=20).grid(row=0, column=0, sticky='N', padx = 20, pady=5)
        label2 = tk.Label(master = window1, text="CrossCorrelationFourier",height=1,width=20).grid(row=0, column=1, sticky='N', padx = 20, pady=5)
        label3 = tk.Label(master = window1, text="cv2.TM_SQDIFF_NORMED",height=1,width=20).grid(row=0, column=2, sticky='N', padx = 20, pady=5)
        label4 = tk.Label(master = window1, text="cv2.TM_CCOEFF",height=1,width=20).grid(row=0, column=3, sticky='N', padx = 20, pady=5)
        label5 = tk.Label(master = window1, text="cv2.TM_CCOEFF_NORMED",height=1,width=20).grid(row=0, column=4, sticky='N', padx = 20, pady=5)
        label6 = tk.Label(master = window1, text="Viola_Jones",height=1,width=20).grid(row=0, column=5, sticky='N', padx = 20, pady=5)
        label7 = tk.Label(master = window1, text="Symmetry + Viola_Jones",height=1,width=20).grid(row=0, column=6, sticky='N', padx = 20, pady=5)
        label8 = tk.Label(master = window1, text="Symmetry",height=1,width=20).grid(row=0, column=7, sticky='N', padx = 20, pady=5)
        im1 = tk.Canvas(master= window1)#, text="TemplateMatchingFourier")
        im1.grid(row=1, column=0, sticky='N', padx = 20, pady=5)
        im2 = tk.Canvas(master= window1)#, text="CrossCorrelationFourier")
        im2.grid(row=1, column=1, sticky='N', padx = 20, pady=5)
        im3 = tk.Canvas(master= window1)#, text="cv2.TM_SQDIFF_NORMED")
        im3.grid(row=1, column=2, sticky='N', padx = 20, pady=5)
        im4 = tk.Canvas(master= window1)#, text="cv2.TM_CCOEFF")
        im4.grid(row=1, column=3, sticky='N', padx = 20, pady=5)
        im5 = tk.Canvas(master= window1)#, text="cv2.TM_CCOEFF_NORMED")
        im5.grid(row=1, column=4, sticky='N', padx = 20, pady=5)
        im6 = tk.Canvas(master= window1)#, text="Viola_Jones")
        im6.grid(row=1, column=5, sticky='N', padx = 20, pady=5)
        im7 = tk.Canvas(master= window1)#, text="Viola_Jones")
        im7.grid(row=1, column=6, sticky='N', padx = 20, pady=5)
        im8 = tk.Canvas(master= window1)#, text="Viola_Jones")
        im8.grid(row=1, column=7, sticky='N', padx = 20, pady=5)

        data = data[0]
        element = data[random.randint(0,data.shape[0]-1)]
        img = Image.fromarray(element)
        img1 = ImageTk.PhotoImage(img)
        template = self.kernel

        t1 = Image.open('t1c.png')
        t2 = Image.open('t2c.png')
        t3 = Image.open('t3c.png')
        t4 = Image.open('t4c.png')
        t5 = Image.open('t5c.png')
        t6 = Image.open('t6c.png')
        t7 = Image.open('t7c.png')
        t8 = Image.open('t8c.png')
        t9 = Image.open('t9c.png')
        t10 = Image.open('t10c.png')
        t11 = Image.open('t11c.png')
        t12 = Image.open('t12c.png')
        t13 = Image.open('t13c.png')
        t14 = Image.open('t14c.png')

        It1 = ImageTk.PhotoImage(t1)
        It2 = ImageTk.PhotoImage(t2)
        It3 = ImageTk.PhotoImage(t3)
        It4 = ImageTk.PhotoImage(t4)
        It5 = ImageTk.PhotoImage(t5)
        It6 = ImageTk.PhotoImage(t6)
        It7 = ImageTk.PhotoImage(t7)
        It8 = ImageTk.PhotoImage(t8)
        It9 = ImageTk.PhotoImage(t9)
        It10 = ImageTk.PhotoImage(t10)
        It11 = ImageTk.PhotoImage(t11)
        It12 = ImageTk.PhotoImage(t12)
        It13 = ImageTk.PhotoImage(t13)
        It14 = ImageTk.PhotoImage(t14)

        tk.Label(master = window1, text="Choose template:",height=1,width=20).grid(row=3, column=3, sticky='N', padx = 20, pady=5)
        bt1 = tk.Button(master= window1, image = It1,command=lambda: self.on_click(t1)).grid(row=4, column=0, sticky='N', padx = 20, pady=5)
        bt2 = tk.Button(master= window1, image = It2,command=lambda: self.on_click(t2)).grid(row=4, column=1, sticky='N', padx = 20, pady=5)
        bt3 = tk.Button(master= window1, image = It3,command=lambda: self.on_click(t3)).grid(row=4, column=2, sticky='N', padx = 20, pady=5)
        bt4 = tk.Button(master= window1, image = It4,command=lambda: self.on_click(t4)).grid(row=4, column=3, sticky='N', padx = 20, pady=5)
        bt5 = tk.Button(master= window1, image = It5,command=lambda: self.on_click(t5)).grid(row=4, column=4, sticky='N', padx = 20, pady=5)
        bt6 = tk.Button(master= window1, image = It6,command=lambda: self.on_click(t6)).grid(row=4, column=5, sticky='N', padx = 20, pady=5)
        bt7 = tk.Button(master= window1, image = It7,command=lambda: self.on_click(t7)).grid(row=4, column=6, sticky='N', padx = 20, pady=5)
        bt8 = tk.Button(master= window1, image = It8,command=lambda: self.on_click(t8)).grid(row=5, column=0, sticky='N', padx = 20, pady=5)
        bt9 = tk.Button(master= window1, image = It9,command=lambda: self.on_click(t9)).grid(row=5, column=1, sticky='N', padx = 20, pady=5)
        bt10 = tk.Button(master= window1, image = It10,command=lambda: self.on_click(t10)).grid(row=5, column=2, sticky='N', padx = 20, pady=5)
        bt11 = tk.Button(master= window1, image = It11,command=lambda: self.on_click(t11)).grid(row=5, column=3, sticky='N', padx = 20, pady=5)
        bt12 = tk.Button(master= window1, image = It12,command=lambda: self.on_click(t12)).grid(row=5, column=4, sticky='N', padx = 20, pady=5)
        bt13 = tk.Button(master= window1, image = It13,command=lambda: self.on_click(t13)).grid(row=5, column=5, sticky='N', padx = 20, pady=5)
        bt14 = tk.Button(master= window1, image = It14,command=lambda: self.on_click(t14)).grid(row=5, column=6, sticky='N', padx = 20, pady=5)

        im1.create_image(0, 0,anchor=tk.NW, image=img1)
        im2.create_image(0, 0,anchor=tk.NW, image=img1)
        im3.create_image(0, 0,anchor=tk.NW, image=img1)
        im4.create_image(0, 0,anchor=tk.NW, image=img1)
        im5.create_image(0, 0,anchor=tk.NW, image=img1)
        im6.create_image(0, 0,anchor=tk.NW, image=img1)
        im7.create_image(0, 0,anchor=tk.NW, image=img1)
        im8.create_image(0, 0,anchor=tk.NW, image=img1)

        labeltemplate = tk.Label(master= window1, text="Current template:",height=1,width=20).grid(row=6, column=3, sticky='N', padx = 20, pady=5)
        imgtemplate = tk.Label(master= window1, text="current template",image = img1)
        imgtemplate.grid(row=7, column=3, sticky='N', padx = 20, pady=5)
        #thread = threading.Thread(target = window1.mainloop)
        #thread.start()

        for i in range(100):
            if not self.running:
                window1.destroy()
                break
            element = data[random.randint(0,data.shape[0]-1)]
            template = self.kernel
            self.change_img(imgtemplate,self.Template)
            img = Image.fromarray(element)
            self.change_img(im1,img)
            self.change_img(im2,img)
            self.change_img(im3,img)
            self.change_img(im4,img)
            self.change_img(im5,img)
            self.change_img(im6,img)
            self.change_img(im7,img)
            self.change_img(im8,img)
            pos1 = TemplateMatchingFourier(element,template)[1][::-1]
            pos2 = CrossCorrelationFourier(element,template)[1][::-1]
            pos3 = cv2.minMaxLoc(cv2.matchTemplate(element,template,cv2.TM_SQDIFF_NORMED))[2]
            pos4 = cv2.minMaxLoc(cv2.matchTemplate(element,template,cv2.TM_CCOEFF))[3]
            pos5 = cv2.minMaxLoc(cv2.matchTemplate(element,template,cv2.TM_CCOEFF_NORMED))[3]
            faces, eyes = self.Viola_Jones(element)
            faceline, eyelines = find_symmetry_lines_viola_jones(faces, eyes,element)
            faceline1, eyelines1 = find_symmetry_simple(element)
            try: faces = faces[0]
            except: pass
            eyel, eyer = [], []
            try: eyel = eyes[0]
            except: pass
            try: eyer = eyes[1]
            except: pass
            r1 = im1.create_rectangle(pos1[0], pos1[1], pos1[0]+template.shape[1], pos1[1]+template.shape[0], fill=None, outline='red', width=3)
            r2 = im2.create_rectangle(pos2[0], pos2[1], pos2[0]+template.shape[1], pos2[1]+template.shape[0], fill=None, outline='red', width=3)
            r3 = im3.create_rectangle(pos3[0], pos3[1], pos3[0]+template.shape[1], pos3[1]+template.shape[0], fill=None, outline='red', width=3)
            r4 = im4.create_rectangle(pos4[0], pos4[1], pos4[0]+template.shape[1], pos4[1]+template.shape[0], fill=None, outline='red', width=3)
            r5 = im5.create_rectangle(pos5[0], pos5[1], pos5[0]+template.shape[1], pos5[1]+template.shape[0], fill=None, outline='red', width=3)
            if len(faces): r6f = im6.create_oval(faces[0], faces[1], faces[0]+faces[2], faces[1]+faces[3], fill=None, outline='red', width=3)
            if len(eyel): r6el = im6.create_oval(faces[0]+eyel[0], faces[1]+eyel[1], faces[0]+eyel[0]+eyel[2], faces[1]+eyel[1]+eyel[3], fill=None, outline='blue', width=3)
            if len(eyer): r6er = im6.create_oval(faces[0]+eyer[0], faces[1]+eyer[1], faces[0]+eyer[0]+eyer[2], faces[1]+eyer[1]+eyer[3], fill=None, outline='blue', width=3)
            if len(faceline): l1 = im7.create_line(faceline[0],faceline[1],faceline[2],faceline[3],fill='blue',width=3)
            if len(eyelines):
                l2, l3 = None, None
                if eyelines[0]:
                    l2 = im7.create_line(eyelines[0][0],eyelines[0][1],eyelines[0][2],eyelines[0][3],fill='blue',width=3)
                if eyelines[1]:
                    l3 = im7.create_line(eyelines[1][0],eyelines[1][1],eyelines[1][2],eyelines[1][3],fill='blue',width=3)
            l4 = im8.create_line(faceline1[0],faceline1[1],faceline1[2],faceline1[3],fill='blue',width=3)
            l5 = im8.create_line(eyelines1[0][0],eyelines1[0][1],eyelines1[0][2],eyelines1[0][3],fill='blue',width=3)
            l6 = im8.create_line(eyelines1[1][0],eyelines1[1][1],eyelines1[1][2],eyelines1[1][3],fill='blue',width=3)
            for i in range(10):
                window1.update()
                time.sleep(2/10)
            try:
                im1.delete(r1)
                im2.delete(r2)
                im3.delete(r3)
                im4.delete(r4)
                im5.delete(r5)
                if len(faces): im6.delete(r6f)
                if len(eyel): im6.delete(r6el)
                if len(eyer): im6.delete(r6er)
                if len(faceline): im7.delete(l1)
                if len(eyelines):
                    if eyelines[0]: im7.delete(l2)
                    if eyelines[1]: im7.delete(l3)
                im8.delete(l4)
                im8.delete(l5)
                im8.delete(l6)
            except: pass
        #window1.mainloop()
        
    def on_click(self,img):
        print("click")
        self.Template = img
        self.kernel = np.array(img.convert('L'),dtype='single');
        self.change_img(self.window.nametowidget("rightframe.templateimage"),img)
        pass


#---------------------------------
application = App()
application.run()
