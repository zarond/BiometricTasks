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

class App:
	def __init__(self):
		self.thread = None
		self.stop = None
		self.root = tk.Tk()
		self.canvas = tk.Canvas(self.root)
		self.root.title("Task 4")
		self.root.protocol("WM_DELETE_WINDOW", self.onClose)
		self.root.bind('<Escape>', lambda e: self.onClose)
		self.root.geometry("1400x700+50+50")
		self.root.resizable(False, True)
		self.canvas.grid(row=0, column=0)
		
		self.frame_settings = tk.Frame(self.canvas, height=100, width=1350)
		self.frame_settings.grid(row=0)
		self.frame_example = tk.Frame(self.canvas, height=600, width=1350)
		self.frame_example.grid(row=1, sticky = 'N')

		self.train_results = tk.Frame(self.canvas)
		self.train_results.grid(row=2,column=0,  pady = 5, sticky = 'NW')
		fig_res1 = tk.Label(self.frame_settings, text="Test results depending on number of images in class")
		fig_res1.grid(row=0, column=10, sticky='N', padx = 20, pady=8)
		self.fig_res2 = tk.Label(self.frame_settings)
		self.fig_res2.grid(row=1, column=10, sticky='N', padx = 20, pady=8,rowspan = 3)
		
		
		lbl1 = tk.Label(self.frame_settings, text="Dataset:")
		lbl1.grid(row=0, column=0, padx = 20, pady = 10)

		self.list = tk.Listbox(self.frame_settings, selectmode = "single", width=30, height=2)
		self.list.grid(row=1, column=0, sticky='N', padx = 20, pady=8)

		m = ["The ORL face database"]
		for each_item in m: 
			self.list.insert("end", each_item)  

		lbl_s = tk.Label(self.frame_settings, text="Scaling factor of images")
		lbl_s.grid(row=1, column=0, sticky='SW', padx = 25, pady = 5)
		self.e_s = tk.Entry(self.frame_settings, width=6)
		self.e_s.grid(row=1, column=0, padx = 25, pady = 5, sticky='SE')

		btn1 = tk.Button(self.frame_settings, text="Load data", command=self.load_data)
		btn1.grid(row=2, column=0, sticky='N', padx = 20, pady = 8)

		self.label_data = tk.Label(self.frame_settings)
		self.label_data.grid(row=3, column=0, sticky='N', padx = 20, pady = 2)

		lbl_i = tk.Label(self.frame_settings, text="Number of images in train")
		lbl_i.grid(row=4, column=0, sticky='SW', padx = 15, pady = 5)
		self.e_i = tk.Entry(self.frame_settings, width=6)
		self.e_i.grid(row=4, column=0, padx = 25, pady = 5, sticky='SE')

		lbl2 = tk.Label(self.frame_settings, text="Voting classificator")
		lbl2.grid(row=0, column=5, padx = 25, pady = 10)

		btn3 = tk.Button(self.frame_settings, text="Compute parameters", command=self.compute)#_start)
		btn3.grid(row=1, column=5, sticky='N', padx = 25, pady = 8)

		lbl9 = tk.Label(self.frame_settings, text="Best results for train:")
		lbl9.grid(row=0, column=7, padx = 25, pady = 8, sticky='N')

		self.label_now_method = tk.Label(self.frame_settings, text="")
		self.label_now_method.grid(row=1, column=8, pady = 10, sticky='NW')  #####

		lbl10 = tk.Label(self.frame_settings, text="Best parameters:")
		lbl10.grid(row=1, column=7, padx = 25, pady = 30, sticky='N')
		self.label_parameter = tk.Label(self.frame_settings, text="")
		self.label_parameter.grid(row=1, column=8, pady = 30, sticky='NW')

		lbl11 = tk.Label(self.frame_settings, text="Best scores:")
		lbl11.grid(row=2, column=7, padx = 25, sticky='N')
		self.label_score = tk.Label(self.frame_settings, text="")
		self.label_score.grid(row=2, column=8, sticky='NW')

		lbl12 = tk.Label(self.frame_settings, text="Best test results:")
		lbl12.grid(row=3, column=7, padx = 25, sticky='N')
		self.label_folds = tk.Label(self.frame_settings, text="")
		self.label_folds.grid(row=3, column=8, sticky='NW')

		self.data = None
		self.number_face_test = [1, 9]
		self.examples = None
		self.parameter = 0
		self.method = None
		
		self.images = [tk.Label(self.frame_example) for i in range(3)]
		for i in range(len(self.images)):
			self.images[i].grid(row = i+1, column = 0, sticky='N', padx=10, pady=2)
		self.features = [[tk.Label(self.frame_example) for i in range(6)] for i in range(3)]
		for i in range(len(self.images)):
			for j in range(len(self.features[0])):
				self.features[i][j].grid(row = i+1, column = j+2, sticky='N', padx=10, pady=2)

		btn4 = tk.Button(self.frame_settings, text="Find one face", command=self.find_one_face)
		btn4.grid(row=3, column=5, sticky='N', padx = 25, pady = 8)
		btn5 = tk.Button(self.frame_settings, text="Test voting classifier", command=self.compute_vote)
		btn5.grid(row=2, column=5, sticky='N', padx = 25, pady = 8)

		lbl6 = tk.Label(self.frame_example, text="Original:")
		lbl6.grid(row=0, column=1, padx = 10, sticky='NW')

		lbl7 = tk.Label(self.frame_example, text="histogram:")
		lbl7.grid(row=0, column=2, padx = 10, sticky='NW')

		lbl8 = tk.Label(self.frame_example, text="DFT:")
		lbl8.grid(row=0, column=3, padx = 10, sticky='NW')

		lbl9 = tk.Label(self.frame_example, text="DCT:")
		lbl9.grid(row=0, column=4, padx = 10, sticky='NW')

		lbl10 = tk.Label(self.frame_example, text="gradient:")
		lbl10.grid(row=0, column=5, padx = 10, sticky='NW')

		lbl11 = tk.Label(self.frame_example, text="scale:")
		lbl11.grid(row=0, column=6, padx = 10, sticky='NW')

		lbl12 = tk.Label(self.frame_example, text="classification example:")
		lbl12.grid(row=0, column=7, padx = 10, sticky='NW')

		self.stop = threading.Event()
		self.work_thread = None

		self.featured_data = None
		self.best_parameter = None
		self.train = None
		self.test = None

	
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

	def feature_example(self):
		if self.data is None:
			tk.messagebox.showinfo("Attention", "Please, select database")
			return
		methods = [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]

		self.examples = [[],[]]
		self.examples[0] = choose_n_from_data(self.test[0],self.test[1], 3)[0]

		for i in range(len(self.examples[0])):
			image = Image.fromarray(self.examples[0][i]*255)
			image = ImageTk.PhotoImage(image)
			self.images[i].configure(image=image)
			self.images[i].image = image
			self.examples[1] = [methods[m](self.examples[0][i],self.best_parameters[m]) for m in range(len(methods))]

			for j in range(len(methods)):
				method = methods[j]
				fig = plt.figure(figsize=(1.1,1.1))
				ax = fig.add_subplot(111)
				if method == histogram_feature:
					hist = self.examples[1][j]
					bins = np.linspace(0, 1, self.best_parameters[j])
					ax.plot(bins, hist)
				if method == dft_feature or method == dct_feature:
					ax.pcolormesh(range(self.examples[1][j].shape[0]),
										range(self.examples[1][j].shape[0]),
										np.flip(self.examples[1][j], 0), cmap="viridis")
				if method == scale_feature:
					image = Image.fromarray(cv2.resize(self.examples[1][j]*255, self.examples[0][0].shape, interpolation = cv2.INTER_AREA))
					ax.imshow(image)
				if method == gradient_feature:
					image_size = self.data[0][0].shape[0]
					ax.plot(range(0, len(self.examples[1][j])), self.examples[1][j])
			
				plt.xticks(color='w')
				plt.yticks(color='w')
				image = ImageTk.PhotoImage(fig2img(fig))
				self.features[i][j].configure(image=image)
				self.features[i][j].image = image
			
				ClassFind = vote_classifier(self.train[0], self.train[1], np.array([self.examples[0][i]]), self.best_parameters,self.featured_data)
				elementFound = (self.train[0][self.train[1]==ClassFind[0]])[0]
				image = Image.fromarray(elementFound*255)
				image = ImageTk.PhotoImage(image)
				self.features[i][5].configure(image=image)
				self.features[i][5].image = image
				
			plt.cla()

	def compute_start(self):
		self.work_thread = threading.Thread(target=self.compute)
		self.work_thread.start()

	def compute(self):
		tic = time.perf_counter()
		self.stop.clear()
		if self.data is None:
			tk.messagebox.showinfo("Attention", "Please, select database")
			return

		images_per_person_in_train = 0
		try: images_per_person_in_train=int(self.e_i.get())
		except: pass
		if images_per_person_in_train<1 or images_per_person_in_train>9: images_per_person_in_train=9
		self.number_face_test[1]=images_per_person_in_train

		x_train, x_test, y_train, y_test = split_data(self.data[0],self.data[1], images_per_person_in_train=images_per_person_in_train,images_per_person_in_test=10-images_per_person_in_train)
		self.train = [x_train,y_train]
		self.test = [ x_test, y_test]
		count = 0

		methods = [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]
		results = [[0, 0, 0] for i in range(len(methods))]

		for m in range(len(methods)):
			for f in [3]:
				if (self.stop.is_set()): return
				res = cross_validation(self.train[0],self.train[1], methods[m], images_per_person=images_per_person_in_train,folds=f)
				if res[0][1] > results[m][1]:
					results[m] = [res[0][0], res[0][1], f]

		#print(results)
		
		str1 = str([i.__name__[:-8] for i in methods])
		#str2 = str(["%.2f" % i[0] for i in results])
		str2 = str(int(results[0][0]))+', '+str(int(results[1][0]))+', '+str(int(results[2][0]))+', '+str(int(results[3][0]))+', '+"%.2f" %results[4][0]
		#str3 = str(["%.2f" % i[1] for i in results])
		str3 = "%.4f" % results[0][1]+', '+"%.4f" % results[1][1]+', '+"%.4f" %results[2][1]+', '+"%.4f" %results[3][1]+', '+"%.4f" %results[4][1]
		#str4 = str([i[2] for i in results])
		self.label_now_method.configure(text = str1)
		self.label_parameter.configure(text = str2)
		self.label_score.configure(text = str3)
		#self.label_folds.configure(text = str4)

		self.featured_data = []
		self.best_parameters = []
		for m in range(len(methods)):
			self.featured_data.append(calculate_feature(self.train[0], methods[m], results[m][0]))
			self.best_parameters.append(results[m][0])

		toc = time.perf_counter()
		print("time computation: ",toc-tic)
		plt.cla()

	def compute_vote(self):
		#A = test_vote_classifier(self.train[0], self.train[1], self.test[0], self.test[1], self.best_parameters, self.featured_data)
		#print("vote accuracy: ",A)
		#self.feature_example()

		sizes = range(int(self.number_face_test[0]), int(self.number_face_test[1]+1))
		#sizes = range(3, 9)
		test_results = [sizes, []]
		tmp = []
		#print(results)
		for size in sizes:
			if (self.stop.is_set()): return
			indices = []
			for i in range(0, self.train[0].shape[0], self.number_face_test[1]):
				indices += list(range(i, i + size))
			tmp = [self.featured_data[0][indices],self.featured_data[1][indices],self.featured_data[2][indices],self.featured_data[3][indices],self.featured_data[4][indices]]
			test_results[1].append(test_vote_classifier(self.train[0][indices], self.train[1][indices], self.test[0], self.test[1], self.best_parameters, tmp))


		self.feature_example()
		self.label_folds.configure(text="{:.6f}".format(max(test_results[1])) + " on " + str(test_results[0][np.argmax(test_results[1])])+" training faces in class")#\nComputation complited in "+"{:.6f}".format(toc-tic)+" seconds")
		
		fig = plt.figure(figsize=(2.5, 2))
		ax = fig.add_subplot(111)
		plt.xticks(test_results[0])
		ax.plot(test_results[0], test_results[1])
		image = ImageTk.PhotoImage(fig2img(fig))
		self.fig_res2.configure(image=image)
		self.fig_res2.image = image

	def find_one_face(self):
		methods = [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]
		window = tk.Toplevel(self.root)
		if not self.best_parameters:
			tk.messagebox.showinfo("Attention", "Error, compute first")
			print("error, compute first")
			return
		if not self.featured_data and self.best_parameters:
			self.featured_data = []
			for m in range(len(methods)):
				self.featured_data.append(calculate_feature(self.train[0], methods[i], results[m][0]))
				self.best_parameters.append(results[m][0])

		indexEl = random.randrange(self.test[0].shape[0])
		element = self.test[0][indexEl]
		ClassFind = vote_classifier(self.train[0], self.train[1], np.array([element]), self.best_parameters,self.featured_data)
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

	def onClose(self):
		print("closing...")
		self.stop.set()
		time.sleep(0.1)
		self.root.destroy()
		#self.root.quit()


application = App()
application.root.mainloop()

