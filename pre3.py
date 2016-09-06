#preprocessing
from PIL import Image # Python Imaging Library (PIL)
import numpy as np
np.set_printoptions(threshold='nan')#otherwise prints the truncated numpy array [0 0 0 ..., 0 0 0]

"""
print maxWidth#227
print maxHeight#212
"""
maxWidth = 227
maxHeight = 212

"""
def normalizeSize(imageFileName):
	image = Image.open(imageFileName)
	imarray = np.array(image)
	image.close()
	imlist = np.array(imarray).tolist()#numpy array to python list
	#print(imarray.shape)#(87, 83) #(row, column) #(height, width)
	#print(imarray.size)#7221

	return imarray.shape
"""

"""limit size"""
maxWidth = 120
maxHeight = 120


for subject in range(1,48):#48
	for imageNum in range(1,121):#121
		try:
			image_FileName = 'trainNO/no-image' + str(subject) + "_" + str(imageNum) + ".png"

			image = Image.open(image_FileName)
			imarray = np.array(image)
			image.close()
			imlist = np.array(imarray).tolist()#numpy array to python list
			#print(imarray.shape)#(87, 83) #(row, column) #(height, width)
			#print(imarray.size)#7221
			
			if imarray.shape[0]<maxHeight or imarray.shape[1]<maxWidth:
				for i in range(0,maxHeight-imarray.shape[0]):#add rows first
					imlist.append([None])
				#print imlist
				for j in range(0, len(imlist)):
					while len(imlist[j])<maxWidth:
						imlist[j].append(None)
			#print len(imlist[0])#227
			#print len(imlist[20
			#print imlist
			"""Save Image"""
			greyim = Image.new('L', (maxWidth,maxHeight))
			# L: (8-bit pixels, black and white)
			npIm =  np.asarray(imlist)
			greyim.putdata(npIm.flatten())
			greyim.save('trainNO2/normalized_no-image'+ str(subject) + "_" + str(imageNum) + ".png" )

		except:
			pass





