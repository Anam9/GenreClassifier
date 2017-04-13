import glob
import os
from tkinter import *

from tkinter.filedialog import askopenfilename
import pygame
import numpy as np
import scipy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import sklearn.linear_model as lm
from sklearn.metrics import confusion_matrix
from matplotlib import pylab
from sklearn import datasets
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

root = Tk()

root.title('Music Classifier')




class start:

	data_fn = ""

	def __init__(self, master):


		frame = Frame(master,height = 200, width = 200)
		frame.pack()

		
		self.file = PhotoImage(file = "gram.png")
		self.background_label = Label(frame, image=self.file)
		self.background_label.place(x=0, y=0, relwidth=1, relheight=1)

		self.background_label.pack()


		self.filename = ""


		self.choose = Button(frame, text = "Browse", command = self.browse, bg = "gray4", fg = "gray99")
		self.choose.pack(fill = X)

		self.disp = Label(frame, text = "   Please choose  ",bg = "gray14", fg = "gray99")
		self.disp.pack(fill = X)

		self.music = Button(frame, text = "Play", command = self.soun, bg = "gray24", fg = "gray99")
		self.music.pack(fill = X)

		self.classify = Button(frame,text = "Convert!", bg = "gray34", fg = "gray99", command = self.convert)
		self.classify.pack(fill = X)

		self.disp2 = Label(frame, text = "   Genre is:   ",bg = "gray44", fg = "gray99")
		self.disp2.pack(fill = X)
		
		

	
		frame2 = Frame(master,height = 200, width = 200)
		frame2.pack()



		self.quit = Button(frame2, text = "Quit", command = frame.quit, relief = GROOVE)
		self.quit.pack(fill = X)
	
	def browse(self):

		
		self.filename = askopenfilename()
		self.disp.config(text = self.filename)	
		Tk().withdraw()

	def soun(self):
		pygame.init()
		sound = pygame.mixer.Sound(self.filename)
		pygame.mixer.Sound.play(sound)
	
	

	
	
	

	# def randomize(dataset, labels):
	#   permutation = np.random.permutation(labels.shape[0])
	#   shuffled_dataset = dataset[permutation,:]
	#   shuffled_labels = labels[permutation]
	#   return shuffled_dataset, shuffled_labels

	# # To randomize
	# train_dataset, train_labels = randomize(train_dataset, train_labels)
	# #test_dataset, test_labels = randomize(test_dataset, test_labels)
	# valid_dataset, valid_labels = randomize(valid_dataset, valid_labels)

		








	

	def convert(self):

		



		genre_list = ["blues", "classical", "country", "disco", "hiphop", "jazz", "metal", "pop", "rock"]
		base_dir = '/home/anam/ML/genres/fft/training'
		X = []
		y = []
		for label, genre in enumerate(genre_list):
			genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
			file_list = glob.glob(genre_dir)
			
			for fn in file_list:
				fft_features = scipy.load(fn)
				X.append(fft_features[:1000])
				y.append(label)
		


		base_dir = '/home/anam/ML/genres/fft/cross_val'
		P = []
		Q = []
		for label, genre in enumerate(genre_list):
			genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
			file_list = glob.glob(genre_dir)
			
			for fn in file_list:
				fft_features = scipy.load(fn)
				P.append(fft_features[:1000])
				Q.append(label)
		




		fn = self.filename
		sample_rate, M = scipy.io.wavfile.read(fn)

		plt.specgram(M, Fs=sample_rate, xextent=(0,30))
		plt.show()
		fft_features = abs(scipy.fft(X)[:1000])

		base_fn, ext = os.path.splitext(fn)	

		data_fn = base_fn + ".fft"
		scipy.save(data_fn, fft_features)

		data_fn = data_fn + ".npy"
		print(data_fn)

		R = []
		S = []
		#for label, genre in enumerate(genre_list):
		#genre_dir = os.path.join(base_dir, genre, "*.fft.npy")
		file_list = glob.glob(data_fn)
		
		
		fft_features = scipy.load(data_fn)
		R = (fft_features[:1000])
		#y.append(label)

		#return np.array(X)

		clff = OneVsRestClassifier(LinearSVC(random_state=0)).fit(X, y)
		z1 = clff.predict(R)
	
		print(z1)

#please change read_fft to a class method 

b = start(root)
root.mainloop()