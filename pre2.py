#preprocessing
from PIL import Image # Python Imaging Library (PIL)
import numpy as np
np.set_printoptions(threshold='nan')#otherwise prints the truncated numpy array [0 0 0 ..., 0 0 0]



def getImageShape(imageFileName):
	image = Image.open(imageFileName)
	imarray = np.array(image)
	image.close()
	imlist = np.array(imarray).tolist()#numpy array to python list
	#print(imarray.shape)#(87, 83) #(row, column) #(height, width)
	#print(imarray.size)#7221

	return imarray.shape

maxWidth=0
maxHeight=0

for subject in range(1,48):
	for imageNum in range(1,121):
		try:
			image_FileName = 'neck_masked_part/image' + str(subject) + "_" + str(imageNum) + ".png"
			returnImageShape = getImageShape(image_FileName)
			#print returnImageShape[0]#87#height
			#print returnImageShape[1]#83#width
			if returnImageShape[0]>maxHeight:
				maxHeight = returnImageShape[0]
				print maxHeight
				print subject
				print imageNum
				print"-----"
			if returnImageShape[1]>maxWidth:
				maxWidth = returnImageShape[1]
				#print maxWidth
				#print subject
				#print imageNum
				#print"-----"
		except:
			pass
"""left out few images by editing the name of the image file becasue they had a weird shape"""

print maxWidth#227
print maxHeight#212
