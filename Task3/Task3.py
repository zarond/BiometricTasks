import numpy as np
import cv2
import tkinter as tk
from PIL import ImageTk, Image, ImageGrab
from tkinter import filedialog
from scipy import signal, spatial, ndimage
import time
import datetime
import os
import threading
from functions import *
import pickle

class App:
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

	def __init__(self):
		self.outputPath = os.path.dirname(os.path.abspath(__file__)) + "/data"
		self.thread = None
		self.stop = None
		self.root = tk.Tk()
		self.canvas = tk.Canvas(self.root, name = 'c')
		self.root.title("Task 3")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.onClose)
		self.root.geometry("1500x800")
		self.root.resizable(True, True)
		self.canvas.grid(row=0, column=0)
		
		self.frame_settings = tk.Frame(self.canvas, height=1450, width=200)
		self.frame_settings.grid(row=0,column = 0, rowspan=2, sticky = 'NW')
		self.frame_example = tk.Frame(self.canvas, height=200, width=800)
		self.frame_example.grid(row=0,column=2, sticky = 'NW')
		self.frame_stats = tk.Frame(self.canvas, name = 'frame_stats', width=1450)
		self.frame_stats.grid(row=1,column=1,  columnspan = 2, pady = 5, sticky = 'NW')
		self.train_results = tk.Frame(self.canvas)
		self.train_results.grid(row=0,column=1,  pady = 5, sticky = 'NW')
		
		lbl1 = tk.Label(self.frame_settings, text="Step 1. Choose dataset scaling")
		lbl1.grid(row=0, column=0, padx = 20, pady = 5, sticky = 'NW')

		self.list = tk.Listbox(self.frame_settings, selectmode = "single", width=30, height=2)
		self.list.grid(row=1, column=0, sticky='N', padx = 20, pady=5)
		m = ["The ORL face database"]
		for each_item in m: 
			self.list.insert("end", each_item)

		lbl_s = tk.Label(self.frame_settings, text="Scaling factor of images")
		lbl_s.grid(row=2, column=0, sticky='SW', padx = 25, pady = 5)
		self.e_s = tk.Entry(self.frame_settings, width=6)
		self.e_s.grid(row=2, column=0, padx = 25, pady = 5, sticky='SE')

		btn1 = tk.Button(self.frame_settings, text="Load data", command=self.load_data)
		btn1.grid(row=3, column=0, sticky='N', padx = 20, pady = 5)

		self.label_data = tk.Label(self.frame_settings)
		self.label_data.grid(row=3, column=0, sticky='NE', padx = 20, pady = 2)

		lbl3 = tk.Label(self.frame_settings, text="Step 2. Choose number of training images in class")
		lbl3.grid(row=4, column=0, padx = 25, pady = 5, sticky = 'NW')

		l1 = tk.Label(self.frame_settings, text="Low range (>=1)")
		l1.grid(row=5, column=0, padx = 25, pady = 0, sticky='NW')
		self.e1 = tk.Entry(self.frame_settings, width=10)
		self.e1.grid(row=5, column=0, padx = 25, pady = 0, sticky='NE')
		l2 = tk.Label(self.frame_settings, text="High range (6<=x<=9)")
		l2.grid(row=7, column=0, padx = 25, pady = 5, sticky='NW')
		self.e2 = tk.Entry(self.frame_settings, width=10)
		self.e2.grid(row=7, column=0, padx = 25, pady = 5, sticky='NE')

		lbl4 = tk.Label(self.frame_settings, text="Step 3. Choose method for calculating features")
		lbl4.grid(row=9, column=0, padx = 25, pady = 5, sticky = 'NW')

		self.list1 = tk.Listbox(self.frame_settings, selectmode = "single", width=20, height=5)
		self.list1.grid(row=10, column=0, sticky='N', padx = 25, pady=5)
		self.methods = ["histogram", "dft", "dct", "scale", "gradient"]
		for each_item in self.methods:
			self.list1.insert("end", each_item)

		btn4 = tk.Button(self.frame_settings, text="Confirm", command=self.set_feature)
		btn4.grid(row=11, column=0, sticky='N', padx = 25, pady = 5)

		self.label_method = tk.Label(self.frame_settings, text="")
		self.label_method.grid(row=13, column=0, padx = 25, pady = 2, sticky='NE')

		lbl2 = tk.Label(self.frame_settings, text="Step 4. Feature example")
		lbl2.grid(row=12, column=0, padx = 25, pady = 5, sticky = 'NW')

		l3 = tk.Label(self.frame_settings, text="Input parameter for feature:")
		l3.grid(row=14, column=0, padx = 25, pady = 8, sticky='NW')
		self.e3 = tk.Entry(self.frame_settings, width=6)
		self.e3.grid(row=14, column=0, padx = 25, pady = 5, sticky='NE')

		btn2 = tk.Button(self.frame_settings, text="Show feature examples", command=self.feature_example)
		btn2.grid(row=15, column=0, sticky='N', padx = 25, pady = 8)

		lbl5 = tk.Label(self.frame_settings, text="Step 5.")
		lbl5.grid(row=16, column=0, padx = 25, pady = 5, sticky = 'NW')

		lbl8 = tk.Label(self.frame_settings, text="Test is calculated using cross validation.\n Best result is chosen")
		lbl8.grid(row=17, column=0, padx = 25, pady = 5, sticky='N')

		btn3 = tk.Button(self.frame_settings, text="Compute", command=self.compute)#_start)
		btn3.grid(row=18, column=0, sticky='N', padx = 25, pady = 5)

		lbl9 = tk.Label(self.train_results, text="Best results for train")
		lbl9.grid(row=12, column=1, padx = 25, pady = 8, sticky='N')

		self.label_now_method = tk.Label(self.train_results, text="")
		self.label_now_method.grid(row=13, column=1, pady = 8, sticky='N')

		lbl10 = tk.Label(self.train_results, text="Best parameter:")
		lbl10.grid(row=14, column=1, padx = 25, pady = 8, sticky='N')
		self.label_parameter = tk.Label(self.train_results, text="")
		self.label_parameter.grid(row=15, column=1, pady = 8, sticky='N')

		lbl11 = tk.Label(self.train_results, text="Best score:")
		lbl11.grid(row=16, column=1, padx = 25, sticky='N')
		self.label_score = tk.Label(self.train_results, text="")
		self.label_score.grid(row=17, column=1, sticky='N')

		lbl12 = tk.Label(self.train_results, text="Best num folds:")
		lbl12.grid(row=18, column=1, padx = 25, sticky='N')
		self.label_folds = tk.Label(self.train_results, text="")
		self.label_folds.grid(row=19, column=1, sticky='N')

		lbl13 = tk.Label(self.train_results, text="Best test result:")
		lbl13.grid(row=20, column=1, padx = 25, sticky='N')
		self.label_best_test_result = tk.Label(self.train_results, text="")
		self.label_best_test_result.grid(row=21, column=1, sticky='N')

		lbl6 = tk.Label(self.frame_example, text="Feature example:")
		lbl6.grid(row=0, column=0, padx = 10,pady = 5, sticky='NW')	

		lbl7 = tk.Label(self.frame_stats, text="Test data computing stats:")
		lbl7.grid(row=0, column=0, padx = 10, sticky='NW')

		self.data = None
		self.number_face_test = [10, 160]
		self.examples = [[], []]
		self.parameter = 0
		self.method = None
		
		self.images = [tk.Label(self.frame_example), tk.Label(self.frame_example), tk.Label(self.frame_example), tk.Label(self.frame_example)]
		for i in range(len(self.images)):
			self.images[i].grid(row = 0, column = i+1, sticky='N', padx=10, pady=2)
		self.features = [tk.Label(self.frame_example), tk.Label(self.frame_example), tk.Label(self.frame_example), tk.Label(self.frame_example)]
		for i in range(len(self.images)):
			self.features[i].grid(row = 1, column = i+1, sticky='N', padx=10, pady=2)

		lll1 = tk.Label(self.frame_stats, name = 'f1', text="")#"train with 2 folds")
		lll1.grid(row=1, column=0, padx = 10)
		lll2 = tk.Label(self.frame_stats, name = 'f2', text="")#"train with 3 folds")
		lll2.grid(row=1, column=1, padx = 10)
		lll3 = tk.Label(self.frame_stats, name = 'f3', text="")#"train with 6 folds")
		lll3.grid(row=1, column=2, padx = 10)
		lll4 = tk.Label(self.frame_stats, text="test with best parameter and different number of training images in class")
		lll4.grid(row=1, column=3, padx = 10)
		self.stats = [tk.Label(self.frame_stats), tk.Label(self.frame_stats), tk.Label(self.frame_stats), tk.Label(self.frame_stats)]
		for i in range(len(self.stats)):
			self.stats[i].grid(row = 2, column = i, sticky='N', padx=10)

		btn4 = tk.Button(self.frame_settings, text="Find one face", command=self.find_one_face)
		btn4.grid(row=19, column=0, sticky='NE', padx = 25, pady = 8)
		btn5 = tk.Button(self.frame_settings, text='auto mode', command=self.auto_mode)
		btn5.grid(row=19, column=0, sticky='NW', padx = 25, pady = 8)
		btn6 = tk.Button(self.frame_settings, text='test and graph', command=self.TestAndGraphAtOnce)
		btn6.grid(row=20, column=0, sticky='NE', padx = 25, pady = 8)
		btn6 = tk.Button(self.frame_settings, text='test and graph realtime', command=self.TestAndGraph)
		btn6.grid(row=20, column=0, sticky='NW', padx = 25, pady = 8)
		btn7 = tk.Button(self.frame_settings, text='save results', command=self.save_results)
		btn7.grid(row=21, column=0, sticky='NE', padx = 25, pady = 8)
		btn8 = tk.Button(self.frame_settings, text='load results', command=self.load_results)
		btn8.grid(row=21, column=0, sticky='NW', padx = 25, pady = 8)

		self.stop = threading.Event()
		self.work_thread = None

		self.featured_data = None
		self.best_parameter = None
		self.train = None
		self.test = None
		self.Folds = None
		self.indices_train = None
		self.indices_test = None
		self.figures_results_data = None
		self.best_acuracy = None
		self.best_num_fold = None
		self.best_time = None
	
	def load_data(self):
		try:
			scale = None
			try:
				scale = float(self.e_s.get())
			except:
				pass
			if not scale or scale<=0.01 or scale>=1:
				scale = 1.0
			self.e_s.delete(0, "end")
			self.e_s.insert(0, scale)
			if self.list.curselection()[0] >= 0:
				if self.list.curselection()[0] == 0:
					self.data = read_data_from_disk()
					if (scale!=1.0):
						self.data[0] = calculate_feature(self.data[0], scale_feature, scale)
					self.label_data.configure(text=str(self.data[0][0].shape))
				#if self.list.curselection()[0] == 1:
				#	self.data = read_data_from_disk()
				#	self.label_data.configure(text="112x92")
		except Exception as e:
			print(e)
			tk.messagebox.showinfo("Attention", "Please, select database")

	def set_feature(self):
		try:
			if self.list1.curselection()[0] >= 0:
				if self.list1.curselection()[0] == 0:
					self.method = histogram_feature
				if self.list1.curselection()[0] == 1:
					self.method = dft_feature
				if self.list1.curselection()[0] == 2:
					self.method = dct_feature
				if self.list1.curselection()[0] == 3:
					self.method = scale_feature
				if self.list1.curselection()[0] == 4:
					self.method = gradient_feature
				self.label_method.configure(text = self.method.__name__)
		except Exception:
			tk.messagebox.showinfo("Attention", "Please, select method")
			return

	def set_parameter(self):
		if self.parameter != 0:
			if self.method == histogram_feature:
				self.parameter = int(self.parameter)
				if self.parameter > 300:
					self.parameter = 300
				if self.parameter < 10:
					self.parameter = 10
			if self.method == dft_feature or self.method == dct_feature:
				self.parameter = int(self.parameter)
				image_size = min(self.data[0][0].shape)
				if self.parameter > image_size - 1:
					self.parameter = image_size - 1
				if self.parameter < 2:
					self.parameter = 2
			if self.method == scale_feature:
				if self.parameter > 1:
					self.parameter = 1
				if self.parameter < 0.05:
					self.parameter = 0.05
			if self.method == gradient_feature:
				self.parameter = int(self.parameter)
				image_size = self.data[0][0].shape[0]
				if self.parameter > int(image_size/2 - 1):
					self.parameter = int(image_size/2 - 1)
				if self.parameter < 2:
					self.parameter = 2

	def feature_example(self):
		if self.data is None:
			tk.messagebox.showinfo("Attention", "Please, select database")
			return
		if self.method is None:
			self.set_feature()
		self.examples[0] = data_for_example(self.data[0])
		try:
			self.parameter = float(self.e3.get())
		except Exception:
			self.parameter = 0

		self.set_parameter()
		self.e3.delete(0, "end")
		self.e3.insert(0, self.parameter)
		if (self.parameter!=0):
			self.examples[1] = [self.method(ex, self.parameter) for ex in self.examples[0]]
		else:
			self.examples[1] = [self.method(ex) for ex in self.examples[0]]

		for i in range(len(self.examples[0])):
			image = Image.fromarray(self.examples[0][i]*255)
			image = ImageTk.PhotoImage(image)
			self.images[i].configure(image=image)
			self.images[i].image = image

			fig = plt.figure(figsize=(1.1,1.1))
			ax = fig.add_subplot(111)
			if self.method == histogram_feature:
				hist = self.examples[1][i]
				bins = np.linspace(0, 1, self.parameter)
				ax.plot(bins, hist)
			if self.method == dft_feature or self.method == dct_feature:
				ax.pcolormesh(range(self.examples[1][i].shape[0]),
									range(self.examples[1][i].shape[0]),
									np.flip(self.examples[1][i], 0), cmap="viridis")
			if self.method == scale_feature:
				image = Image.fromarray(cv2.resize(self.examples[1][i]*255, self.examples[0][i].shape, interpolation = cv2.INTER_AREA))
				ax.imshow(image)
			if self.method == gradient_feature:
				image_size = self.data[0][0].shape[0]
				ax.plot(range(0, len(self.examples[1][i])), self.examples[1][i])
			
			plt.xticks(color='w')
			plt.yticks(color='w')
			image = ImageTk.PhotoImage(fig2img(fig))
			self.features[i].configure(image=image)
			self.features[i].image = image
			
			plt.cla()
			plt.clf()
			plt.close()
		#plt.cla()
		#plt.clf()
		#plt.close()

	def compute_start(self):
		self.work_thread = threading.Thread(target=self.compute)
		self.work_thread.start()

	def compute(self):
		tic = time.perf_counter()
		self.stop.clear()
		if self.data is None:
			tk.messagebox.showinfo("Attention", "Please, select database")
			return
		try:
			self.number_face_test = [int(self.e1.get()), int(self.e2.get())]
		except Exception:
			tk.messagebox.showinfo("Attention", "Please, input integer numbers in range [1, 9]")
			return
		if self.number_face_test[0] < 1:
			self.number_face_test[0] = 1
			self.e1.delete(0, "end")
			self.e1.insert(0, self.number_face_test[0])
		if self.number_face_test[1] > 9 or self.number_face_test[1] < 6:
			self.number_face_test[1] = max(6,min(self.number_face_test[1],9))
			self.e2.delete(0, "end")
			self.e2.insert(0, self.number_face_test[1])
		if self.method is None:
			self.set_feature()
		results = [0, 0, 0]
		x_train, x_test, y_train, y_test, self.indices_train, self.indices_test= split_data(self.data[0],self.data[1], images_per_person_in_train=self.number_face_test[1],images_per_person_in_test=10-self.number_face_test[1])
		self.train = [x_train,y_train]
		self.test = [ x_test, y_test]
		self.Folds = [2, 3, 6]
		if self.number_face_test[1] == 9: self.Folds = [3, 4, 9]
		if self.number_face_test[1] == 8: self.Folds = [2, 4, 8]
		if self.number_face_test[1] == 7: self.Folds = [2, 3, 7]
		if self.number_face_test[1] == 6: self.Folds = [2, 3, 6]
		self.root.nametowidget("c.frame_stats.f1").config( text = "train with "+str(self.Folds[0])+" folds")
		self.root.nametowidget("c.frame_stats.f2").config( text = "train with "+str(self.Folds[1])+" folds")
		self.root.nametowidget("c.frame_stats.f3").config( text = "train with "+str(self.Folds[2])+" folds")
		figs = plt.figure(figsize=(9, 3))
		axes = [figs.add_subplot(1,3,1),figs.add_subplot(1,3,2),figs.add_subplot(1,3,3)]
		count=0
		self.figures_results_data = []
		for f in self.Folds:
			if (self.stop.is_set()): return
			res = cross_validation(self.train[0],self.train[1], self.method, images_per_person=self.number_face_test[1],folds=f)
			if res[0][1] > results[1]:
				results = [res[0][0], res[0][1], f]
			#plt.rcParams["font.size"] = "5"
			fig = plt.figure(figsize=(2.5, 2))
			ax = fig.add_subplot(111)
			ax.plot(res[1][0], res[1][1])
			ax.set_ylim(ymin=0)
			ax.set_xlabel('parameter value')
			ax.set_ylabel('classification accuracy')
			ax.yaxis.set_ticks(np.linspace(0, 1, 11))
			
			axes[count].plot(res[1][0], res[1][1])
			axes[count].set_ylim(ymin=0)
			axes[count].yaxis.set_ticks(np.linspace(0, 1, 11))
			axes[count].set_title("train with "+str(f)+" folds")

			axes[count].set_xlabel('parameter value')
			axes[count].set_ylabel('classification accuracy')

			image = ImageTk.PhotoImage(fig2img(fig))
			self.stats[count].configure(image=image)
			self.stats[count].image = image
			count += 1

			self.figures_results_data.append([res[1][0], res[1][1]])

			ax.cla()
			fig.clf()
			plt.close(fig)

		figs.show()

		self.best_acuracy = results[1]
		self.best_num_fold = results[2]
		
		self.label_now_method.configure(text = str(self.method.__name__[:-8]))
		self.label_parameter.configure(text = "{:.6f}".format(results[0]))
		self.label_score.configure(text = "{:.6f}".format(results[1]))
		self.label_folds.configure(text = str(results[2]))

		sizes = range(int(self.number_face_test[0]), int(self.number_face_test[1]+1))
		test_results = [sizes, []]
		self.featured_data = calculate_feature(self.train[0], self.method, results[0])
		self.best_parameter = results[0]
		print(results)
		for size in sizes:
			if (self.stop.is_set()): return
			#tmp = choose_n_from_data(self.test[0],self.test[1], size)
			#test_results[1].append(test_classifier(self.featured_data, self.train[1], tmp[0], tmp[1], self.method, results[0]))
			indices = []
			for i in range(0, self.train[0].shape[0], self.number_face_test[1]):
				indices += list(range(i, i + size))
			test_results[1].append(test_classifier(self.featured_data[indices], self.train[1][indices], self.test[0], self.test[1], self.method, results[0]))
		#plt.rcParams["font.size"] = "5"
		fig = plt.figure(figsize=(2.5, 2))
		ax = fig.add_subplot(111)
		ax.plot(test_results[0], test_results[1])
		ax.xaxis.set_ticks(test_results[0])
		ax.set_ylim(ymin=0)
		plt.xlabel('number of training images in class')
		plt.ylabel('classification accuracy')
		ax.yaxis.set_ticks(np.linspace(0, 1, 11))
		image = ImageTk.PhotoImage(fig2img(fig))
		self.stats[count].configure(image=image)
		self.stats[count].image = image

		self.figures_results_data.append([test_results[0], test_results[1]])

		plt.cla()
		plt.clf()
		plt.close()

		fig = plt.figure(figsize=(7, 3))
		ax = fig.add_subplot(111)
		ax.plot(test_results[0], test_results[1])
		ax.xaxis.set_ticks(test_results[0])
		ax.set_ylim(ymin=0)
		plt.xlabel('number of training images in class')
		plt.ylabel('classification accuracy')
		ax.yaxis.set_ticks(np.linspace(0, 1, 11))

		toc = time.perf_counter()
		print("time computation: ",toc-tic)
		self.best_time = toc - tic
		self.label_best_test_result.configure(text="{:.6f}".format(max(test_results[1])) + " on " + str(test_results[0][np.argmax(test_results[1])])+" training faces in class\nComputation complited in "+"{:.6f}".format(toc-tic)+" seconds")

		ax.set_title("test with best parameter and different number of training images in class")
		fig.show()
		#plt.cla()
		#plt.clf()
		#plt.close()


	def find_one_face(self):
		window = tk.Toplevel(self.root)	
		indexEl = random.randrange(self.test[0].shape[0])
		element = self.test[0][indexEl]
		if not self.featured_data.size:
			tk.messagebox.showinfo("Attention", "Error, compute first")
			print("error, compute first")
			return
		ClassFind = classifier(self.featured_data, self.train[1], np.array([element]), self.method, self.best_parameter)
		print("original class:",self.test[1][indexEl],". class found: ", int(ClassFind[0]))
		if not ClassFind: print("error")
		elementFound = (self.train[0][self.train[1]==ClassFind[0]])[0]
		image1 = ImageTk.PhotoImage(Image.fromarray(element*256))
		image2 = ImageTk.PhotoImage(Image.fromarray(elementFound*256))
		l1 = tk.Label(window, text="train image from class "+str(self.test[1][indexEl])).pack()
		l2 = tk.Label(window, image = image1).pack()
		l3 = tk.Label(window, text="train image from class "+str(int(ClassFind[0]))).pack()
		l4 = tk.Label(window, image = image2).pack()
		window.mainloop()
		pass

	def auto_mode(self):
		window1 = tk.Toplevel(self.root)
		window1.protocol("WM_DELETE_WINDOW", self.close_window)
		self.running = True
		label1 = tk.Label(master = window1, text="Classification test:",height=1,width=20).grid(row=0, column=0, sticky='N', padx = 20, pady=5)
		im1 = tk.Label(master= window1)
		im1.grid(row=2, column=0, sticky='N', padx = 20, pady=5)
		im2 = tk.Label(master= window1)
		im2.grid(row=2, column=1, sticky='N', padx = 20, pady=5)
		im3 = tk.Label(master= window1)
		im3.grid(row=2, column=2, sticky='N', padx = 20, pady=5)
		im4 = tk.Label(master= window1)
		im4.grid(row=2, column=3, sticky='N', padx = 20, pady=5)
		rim1 = tk.Label(master= window1)
		rim1.grid(row=4, column=0, sticky='N', padx = 20, pady=5)
		rim2 = tk.Label(master= window1)
		rim2.grid(row=4, column=1, sticky='N', padx = 20, pady=5)
		rim3 = tk.Label(master= window1)
		rim3.grid(row=4, column=2, sticky='N', padx = 20, pady=5)
		rim4 = tk.Label(master= window1)
		rim4.grid(row=4, column=3, sticky='N', padx = 20, pady=5)

		train = self.train[0]
		test = self.test[0]
		method = self.method
		best_parameter = self.best_parameter

		tk.Label(master = window1, text="Test images:",height=1,width=20).grid(row=1, column=0, sticky='N', padx = 20, pady=5)

		tk.Label(master= window1, text="Training images:",height=1,width=20).grid(row=3, column=0, sticky='N', padx = 20, pady=5)
		#thread = threading.Thread(target = window1.mainloop)
		#thread.start()

		for i in range(100):
			if not self.running:
				window1.destroy()
				break
			ind = random.sample(range(0, test.shape[0]), 4)
			elements = test[ind]
			img1 = Image.fromarray(elements[0]*256)
			img2 = Image.fromarray(elements[1]*256)
			img3 = Image.fromarray(elements[2]*256)
			img4 = Image.fromarray(elements[3]*256)
			self.change_img(im1,img1)
			self.change_img(im2,img2)
			self.change_img(im3,img3)
			self.change_img(im4,img4)
			ClassFind = classifier(self.featured_data, self.train[1], np.array(elements), method, best_parameter)
			#print("original class:",self.test[1][indexEl],". class found: ", int(ClassFind[0]))
			if not np.all(ClassFind): print("error")
			boolFound = (self.test[1][ind]==ClassFind)
			if boolFound[0]: rim1.config(bg="green")
			else: rim1.config(bg="red")
			if boolFound[1]: rim2.config(bg="green")
			else: rim2.config(bg="red")
			if boolFound[2]: rim3.config(bg="green")
			else: rim3.config(bg="red")
			if boolFound[3]: rim4.config(bg="green")
			else: rim4.config(bg="red")
			elementFound1 = (train[self.train[1]==ClassFind[0]])[0]
			elementFound2 = (train[self.train[1]==ClassFind[1]])[0]
			elementFound3 = (train[self.train[1]==ClassFind[2]])[0]
			elementFound4 = (train[self.train[1]==ClassFind[3]])[0]
			rimg1 = Image.fromarray(elementFound1*256)
			rimg2 = Image.fromarray(elementFound2*256)
			rimg3 = Image.fromarray(elementFound3*256)
			rimg4 = Image.fromarray(elementFound4*256)
			self.change_img(rim1,rimg1)
			self.change_img(rim2,rimg2)
			self.change_img(rim3,rimg3)
			self.change_img(rim4,rimg4)

			if not self.running:
				window1.destroy()
				break
			
			for i in range(10):
				window1.update()
				time.sleep(2/10)

		#window1.mainloop()

	def close_window(self):
		self.running = False  # turn off while lo
		#window1.destroy()
		#window1.quit()

	def TestAndGraph(self):
		from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
		window1 = tk.Toplevel(self.root)
		C = tk.Canvas(master=window1)
		C.grid(row=0,column=0)
		test = self.test.copy()
		test = shuffle_data(test[0],test[1])
		Total = 0
		correct = 0
		#plt.ion()
		fig = plt.figure()
		ax = fig.add_subplot(111)
		result = np.empty(test[0].shape[0])
		ax.plot([0]*10)
		ax.set_title("classification of test data")
		ax.set_ylim(ymin=0)
		plt.xlabel('number of testing images')
		plt.ylabel('classification accuracy')
		#plt.show()
		#fig.canvas.draw()
		canvas = FigureCanvasTkAgg(fig, C)
		canvas.get_tk_widget().grid(row=0,column=0)#pack(side=tk.TOP, fill=tk.BOTH, expand=1)
		canvas.draw()
		for i in range(test[0].shape[0]):
			ax.cla()
			el =  test[0][i]
			ClassFind = classifier(self.featured_data, self.train[1], np.array([el]), self.method, self.best_parameter)
			boolFound = (test[1][i]==ClassFind[0])
			if boolFound: correct=correct+1
			Total=Total+1
			result[i] = correct/Total
			ax.plot(result[:i+1])
			ax.set_ylim(ymin=0)
			ax.set_xlabel('number of testing images')
			ax.set_ylabel('classification accuracy')
			canvas.draw()
			#time.sleep(0.01)
			window1.update()
		print("classification complete, ",correct,"/",Total," correct")
		tk.messagebox.showinfo("Attention", "classification complete, "+ str(correct)+"/"+str(Total)+" correct")
		window1.mainloop()

	def TestAndGraphAtOnce(self):
		test = self.test.copy()
		test = shuffle_data(test[0],test[1])
		Total = 0
		correct = 0
		fig = plt.figure()
		ax = fig.add_subplot(111)
		result = np.empty(test[0].shape[0])
		ax.set_title("classification of test data")
		ClassFind = classifier(self.featured_data, self.train[1], test[0], self.method, self.best_parameter)
		boolFound = (test[1]==ClassFind)
		for i in range(test[0].shape[0]):
			b = boolFound[i]
			if b: correct=correct+1
			Total=Total+1
			result[i] = correct/Total
		print("classification complete, ",correct,"/",Total," correct")
		ax.plot(result)
		plt.xlabel('number of testing images')
		plt.ylabel('classification accuracy')
		ax.set_ylim(ymin=0)
		tk.messagebox.showinfo("Attention", "classification complete, "+ str(correct)+"/"+str(Total)+" correct")
		plt.show()

	def save_results(self):
		obj = [self.method,self.best_parameter,self.number_face_test, self.indices_train, 
		 self.indices_test, self.figures_results_data,	self.best_acuracy, self.best_num_fold, self.best_time]
		#f = open('out.data','wb')
		f = filedialog.asksaveasfilename()
		f = open(f,'wb')
		pickle.dump(obj,f)
		pass

	def load_results(self):
		f = filedialog.askopenfilename()
		if f=='': return
		f = open(f,'rb')
		obj = pickle.load(f)
		self.method = obj[0]
		self.best_parameter = obj[1]
		self.number_face_test = obj[2]
		self.indices_train = obj[3]
		self.indices_test = obj[4]
		x_train = self.data[0][self.indices_train]
		x_test = self.data[0][self.indices_test]
		y_train = self.data[1][self.indices_train]
		y_test = self.data[1][self.indices_test]
		self.train = [x_train,y_train]
		self.test = [x_test,y_test]
		self.featured_data = calculate_feature(self.train[0], self.method, self.best_parameter)
		self.e1.delete(0, "end")
		self.e1.insert(0, self.number_face_test[0])
		self.e2.delete(0, "end")
		self.e2.insert(0, self.number_face_test[1])
		self.figures_results_data = obj[5]
		self.best_acuracy = obj[6]
		self.best_num_fold = obj[7]
		self.best_time = obj[8]
		self.Folds = [2, 3, 6]
		if self.number_face_test[1] == 9: self.Folds = [3, 4, 9]
		if self.number_face_test[1] == 8: self.Folds = [2, 4, 8]
		if self.number_face_test[1] == 7: self.Folds = [2, 3, 7]
		if self.number_face_test[1] == 6: self.Folds = [2, 3, 6]
		figs = plt.figure(figsize=(9, 3))
		axes = [figs.add_subplot(1,3,1),figs.add_subplot(1,3,2),figs.add_subplot(1,3,3)]
		count=0
		self.root.nametowidget("c.frame_stats.f1").config( text = "train with "+str(self.Folds[0])+" folds")
		self.root.nametowidget("c.frame_stats.f2").config( text = "train with "+str(self.Folds[1])+" folds")
		self.root.nametowidget("c.frame_stats.f3").config( text = "train with "+str(self.Folds[2])+" folds")
		for f in self.Folds:
			fig = plt.figure(figsize=(2.5, 2))
			ax = fig.add_subplot(111)
			ax.plot(self.figures_results_data[count][0],self.figures_results_data[count][1])
			ax.set_ylim(ymin=0)
			ax.set_xlabel('parameter value')
			ax.set_ylabel('classification accuracy')
			ax.yaxis.set_ticks(np.linspace(0, 1, 11))
			axes[count].plot(self.figures_results_data[count][0],self.figures_results_data[count][1])
			axes[count].set_ylim(ymin=0)
			axes[count].yaxis.set_ticks(np.linspace(0, 1, 11))
			axes[count].set_title("train with "+str(f)+" folds")
			axes[count].set_xlabel('parameter value')
			axes[count].set_ylabel('classification accuracy')
			image = ImageTk.PhotoImage(fig2img(fig))
			self.stats[count].configure(image=image)
			self.stats[count].image = image
			count += 1
			ax.cla()
			fig.clf()
			plt.close(fig)
		figs.show()
		self.label_now_method.configure(text = str(self.method.__name__[:-8]))
		self.label_parameter.configure(text = "{:.6f}".format(self.best_parameter))
		self.label_score.configure(text = "{:.6f}".format(self.best_acuracy))##############
		self.label_folds.configure(text = str(self.best_num_fold))#############

		fig = plt.figure(figsize=(2.5, 2))
		ax = fig.add_subplot(111)
		ax.plot(self.figures_results_data[3][0],self.figures_results_data[3][1])
		ax.xaxis.set_ticks(self.figures_results_data[3][0])
		ax.set_ylim(ymin=0)
		plt.xlabel('number of training images in class')
		plt.ylabel('classification accuracy')
		ax.yaxis.set_ticks(np.linspace(0, 1, 11))
		image = ImageTk.PhotoImage(fig2img(fig))
		self.stats[count].configure(image=image)
		self.stats[count].image = image
		plt.cla()
		plt.clf()
		plt.close()
		fig = plt.figure(figsize=(7, 3))
		ax = fig.add_subplot(111)
		ax.plot(self.figures_results_data[3][0],self.figures_results_data[3][1])
		ax.xaxis.set_ticks(self.figures_results_data[3][0])
		ax.set_ylim(ymin=0)
		plt.xlabel('number of training images in class')
		plt.ylabel('classification accuracy')
		ax.yaxis.set_ticks(np.linspace(0, 1, 11))
		self.label_best_test_result.configure(text="{:.6f}".format(max(self.figures_results_data[3][1])) + " on " + str(self.figures_results_data[3][0][np.argmax(self.figures_results_data[3][1])])+" training faces in class\nComputation complited in "+"{:.6f}".format(self.best_time)+" seconds")
		ax.set_title("test with best parameter and different number of training images in class")
		fig.show()
		

	def onClose(self):
		print("closing...")
		self.stop.set()
		time.sleep(0.1)
		self.root.destroy()
		#self.root.quit()


application = App()
application.root.mainloop()
