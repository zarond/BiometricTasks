import numpy as np
import matplotlib.pyplot as plt
import random
import os
from scipy import fftpack ,spatial
import cv2
from PIL import Image
import io

def fig2img(fig):
    """Convert a Matplotlib figure to a PIL Image and return it"""
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img

def histogram_feature(image:np.array, param = 30):
    hist, bins = np.histogram(image, bins=np.linspace(0, 1, param+1))
    return hist#[hist, bins]

def dft_feature(image:np.array, size = 13):
    r = np.fft.fft2(image)
    r = r[0:size, 0:size]
    return np.abs(r)

def dct_feature(image:np.array, size = 13):
    r = fftpack.dct(image, axis=1)
    r = fftpack.dct(r, axis=0)
    r = r[0:size, 0:size]
    return r

def gradient_feature(image:np.array, n = 2):
    size = image.shape[0]//n
    result = np.empty(size-1)
    for i in range(size-1):
         result[i] = np.sum(np.square(image[i*n:i*n+n,:]-np.flip(image[i*n+n:i*n+2*n,:],axis=0)))
    return result

def scale_feature(image:np.array, scale = 0.35):
	h = image.shape[0]
	w = image.shape[1]
	new_size = (int(w * scale), int(h * scale))
	return cv2.resize(image, new_size, interpolation = cv2.INTER_AREA)

def read_data_from_disk():
	faces = []
	target = []
	folder = os.path.dirname(os.path.abspath(__file__)) + "/orl_faces/s"
	for i in range(1, 41):
		for j in range(1, 11):
			image = cv2.cvtColor(cv2.imread(folder + str(i) + "/" + str(j) + ".pgm"), cv2.COLOR_BGR2GRAY)
			faces.append(image/255)
			target.append(i)
	return [np.array(faces), np.array(target)]

def data_for_example(data:np.array):
	return [data[12], data[14], data[25], data[66]]

def shuffle_data(data:np.array,target:np.array):
	return choose_n_from_data(data,target, data.shape[0])

def split_data(data:np.array,target:np.array, images_per_person_in_train=5, images_per_person_in_test=1, images_per_person = 10):
	images_all = data.shape[0]
	images_per_person_in_train = max(min(images_per_person_in_train,9),1)
	images_per_person_in_test = max(min(images_per_person_in_test,9),1)
	if (images_per_person_in_train+images_per_person_in_test>10): images_per_person_in_test=10-images_per_person_in_train

	indices_train = []
	indices_test = []

	for i in range(0, images_all, images_per_person):
		indices = range(i, i + images_per_person)
		indices_train += random.sample(indices, images_per_person_in_train)
		indices_test += random.sample(set(indices) - set(indices_train), images_per_person_in_test)

	x_train = data[indices_train]
	x_test = data[indices_test]
	y_train = target[indices_train]
	y_test = target[indices_test]
	
	return x_train, x_test, y_train, y_test

def choose_n_from_data(data:np.array,target:np.array, number):
	indexes = random.sample(range(0, data.shape[0]), number)
	return data[indexes], target[indexes]

def calculate_feature(data:np.array, method, parameter):
	#result = np.empty(data.shape[0])
	result = []
	for i in range(data.shape[0]):
		#result[i]=method(data[i], parameter)
		result.append(method(data[i], parameter))
	return np.array(result)

def distance(a:np.array, b:np.array):
	return np.linalg.norm(a - b)

def classifier(featured_data:np.array, target:np.array, new_elements:np.array, method, parameter):
	if method not in [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]:
		return []
	#featured_data = calculate_feature(data, method, parameter)
	featured_elements = calculate_feature(new_elements, method, parameter)
	result = np.empty(new_elements.shape[0])
	definescipy = False#True
	if not definescipy:
		for i in range(featured_elements.shape[0]):
			min_el = [2**50, -1]
			for j in range(featured_data.shape[0]):
				dist = distance(featured_elements[i], featured_data[j])
				if dist < min_el[0]:
					min_el = [dist, j]
			if min_el[1] < 0:
				result[i]=0
			else:
				result[i]=target[min_el[1]]
	else:
		featured_data1 = np.reshape(featured_data,[featured_data.shape[0],-1])
		featured_elements1 = np.reshape(featured_elements,[featured_elements.shape[0],-1])
		result = target[np.argmin(spatial.distance_matrix(featured_elements1,featured_data1),axis=1)]
	return result

def test_classifier(featured_data:np.array, target:np.array, test_elements:np.array, test_targets:np.array, method, parameter):
	if method not in [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]:
		return []
	answers = classifier(featured_data, target, test_elements, method, parameter)
	correct_answers = np.count_nonzero(answers==test_targets)
	return correct_answers/test_targets.shape[0]

def vote_classifier(data:np.array,  target:np.array, new_elements:np.array, parameters, featured_data=None):
	methods = [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]
	res = [0]*len(methods)
	if not featured_data:
		for i in range(len(methods)):
			featured_data = calculate_feature(data, methods[i], parameters[i])
			res[i] = classifier(featured_data, target, new_elements, methods[i], parameters[i])
	else:
		for i in range(len(methods)):
			res[i] = classifier(featured_data[i], target, new_elements, methods[i], parameters[i])
	tmp = []
	for i in range(new_elements.shape[0]):
		set_of_answers = {}
		for method in res:
			t = method[i]
			if t in set_of_answers:
				set_of_answers[t] += 1
			else:
				set_of_answers[t] = 1
		best_answer = sorted(set_of_answers, key=set_of_answers.get, reverse=True)[0]
		tmp.append(best_answer)
	return np.array(tmp)

def test_vote_classifier(data:np.array, target:np.array, test_data:np.array, test_targets:np.array, parameters, featured_data=None):
	answers = vote_classifier(data, target, test_data, parameters, featured_data)
	correct_answers = np.count_nonzero(answers==test_targets)
	return correct_answers/test_targets.shape[0]

def teach_parameter(data:np.array, target:np.array, test_elements:np.array, test_targets:np.array, method):
	if method not in [histogram_feature, dft_feature, dct_feature, gradient_feature, scale_feature]:
		return []
	image_size = min(data[0].shape)
	param = (0, 0, 0)
	if method == histogram_feature:
		param = (10, 300, 3)
	if method == dft_feature or method == dct_feature:
		param = (2, image_size, 1)
	if method == gradient_feature:
		param = (2, int(data[0].shape[0]/2 - 1), 1)
	if method == scale_feature:
		param = (0.05, 1, 0.05)
	
	best_param = param[0]
	featured_data = calculate_feature(data, method, best_param)
	classf = test_classifier(featured_data, target, test_elements, test_targets, method, best_param)
	stat = [[best_param], [classf]]

	for i in np.arange(param[0] + param[2], param[1], param[2]):
		featured_data = calculate_feature(data, method, i)
		new_classf = test_classifier(featured_data, target, test_elements, test_targets, method, i)
		stat[0].append(i)
		stat[1].append(new_classf)
		if new_classf > classf:
			classf = new_classf
			best_param = i
	
	return [best_param, classf], stat

def cross_validation(data:np.array, target:np.array, method, images_per_person, folds=3):
	if folds < 3:
		folds = 3
	#per_fold = int(data.shape[0]/folds)
	fold_step = int(images_per_person/folds)
	per_fold = int(images_per_person//folds+(images_per_person%folds>0))
	results = []
	for step in range(0, folds):
		print("fold " + str(step))
		indices_test,indices_train=[],[]
		for i in range(0, data.shape[0], images_per_person):
			indices = range(i, i + images_per_person)
			if step==folds-1:
				indices_test += indices[-per_fold:]
			else:
				indices_test += indices[step*fold_step:step*fold_step+per_fold]
			indices_train += set(indices) - set(indices_test)

		x_train = data[indices_train]
		x_test = data[indices_test]
		y_train = target[indices_train]
		y_test = target[indices_test]
		results.append(teach_parameter(x_train, y_train, x_test, y_test, method))
	res = results[0]
	for element in results[1:]:
		best = element[0]
		stat = element[1]
		res[0][0] += best[0]
		res[0][1] += best[1]
		for i in range(len(stat[1])):
			res[1][1][i] += stat[1][i]
	res[0][0] /= folds
	if method != scale_feature:
		res[0][0] = int(res[0][0])
	res[0][1] /= folds
	for i in range(len(res[1][1])):
		res[1][1][i] /= folds
	return res

