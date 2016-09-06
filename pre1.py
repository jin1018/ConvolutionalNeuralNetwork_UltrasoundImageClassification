#preprocessing
from PIL import Image # Python Imaging Library (PIL)
import numpy as np
np.set_printoptions(threshold='nan')#otherwise prints the truncated numpy array [0 0 0 ..., 0 0 0]

"""Mask image"""
def getDimensionsOfMask(imageFileName):
	"""return indicesOfMask, hightOfMask, widthOfMask, widthMin, widthMax in order in an array"""

	image = Image.open(imageFileName)
	maskInfo=[]
	#im.show()#shows the image as bmp file
	imarray = np.array(image)
	image.close()
	#imlist = np.array(imarray).tolist()#numpy array to python list
	#print(imarray.shape)#(420,580) #Tuple of array dimensions #row * column
	#print(imarray.size)#420 * 580 =  243600 #Number of elements in the array
	#print(imarray)
	#print imlist

	searchval = 255
	indicesOfMask = np.where(imarray == searchval)#2d array: 1st array is index of rows, 2nd array is index of columns
	#indicesOfMask = np.where(imarray == searchval)[0]#row
	#indicesOfMask = np.where(imarray == searchval)[1]#column
	#print(indicesOfMask)

	#get hight of mask: number of rows
	hightOfMask = (indicesOfMask[0][0] - indicesOfMask[0][len(indicesOfMask[0])-1])*(-1)+1
	#print hightOfMask

	#get width of mask: number of of columns
	widthOfMask= np.ptp(indicesOfMask[1]) + 1 #ptp(peak to peak): max-min
	#print widthOfMask
	widthMin = np.min(indicesOfMask[1])
	widthMax = np.max(indicesOfMask[1])
	#print widthMax
	#print widthMin

	maskInfo.append(indicesOfMask)
	maskInfo.append(hightOfMask)#number of rows
	maskInfo.append(widthOfMask)#number of columns
	maskInfo.append(widthMin)
	maskInfo.append(widthMax)

	return maskInfo


"""Ultrasound Image"""
def getDimensionsOfNeckImage(array_dimensionsOfMask, nameOfFile_neck):
	"""
	return: multi-dimensioanl array of Selected Images from Mask and NonSelected Images(to use for input data)
	"""
	image_neck = Image.open(nameOfFile_neck)
	imarray_neck = np.array(image_neck)
	image_neck.close()
	#print(imarray_neck.shape)#(420,580) #Tuple of array dimensions #row * column
	#print(imarray_neck.size)#420 * 580 =  243600 #Number of elements in the array
	#print(imarray_neck)

	#select the pixels correspondsing to the mask.
	"""
	ToTest:
	If there is no selection, should fill up the pixels with NULL? 
	Or should I leave it as it is(black pixels)?
	"""
	#initialize 2d array for iamge
	selectedImage = [[None for y in range(array_dimensionsOfMask[2])] for x in range(array_dimensionsOfMask[1])]
	#initialize 2d array for NonCell image
	nonSelectedImage = [[None for y in range(array_dimensionsOfMask[2])] for x in range(array_dimensionsOfMask[1])]

	#print(len(selectedImage))#87 #rows
	#print(len(selectedImage[0]))#83 #columns

	startIndexOfMask_x = array_dimensionsOfMask[0][0][0] #initialze currentRow
	#print len(array_dimensionsOfMask[0][0])#4616

	for i in range(0,len(array_dimensionsOfMask[0][0])):#range: include 0, stop before len(array_dimensionsOfMask[0])
		x=array_dimensionsOfMask[0][0][i] #row
		y=array_dimensionsOfMask[0][1][i] #column
		neckImagePixelValue = imarray_neck[x,y]
		""" debug
		print x
		print y 
		print neckImagePixelValue
		print x-startIndexOfMask_x
		print y-array_dimensionsOfMask[3]
		print array_dimensionsOfMask[3]
		print startIndexOfMask_x
		print("------")
		"""
		selectedImage[x-startIndexOfMask_x][y-array_dimensionsOfMask[3]] = neckImagePixelValue
		npSelectedImage =  np.asarray(selectedImage)

		"""Get Non-Cell Images for training set input"""
		"""
		try:
			x=x-10
			y=y-10
			nonNeckPixelValue=imarray_neck[x,y]
			nonSelectedImage[x-startIndexOfMask_x][y-array_dimensionsOfMask[3]] = nonNeckPixelValue
			npNonSelectedImage = np.asarray(nonSelectedImage)
		except:
			print("not working")
			pass
		"""

		"""Can change location of the non-cell image here"""
		nonNeckPixelValue=imarray_neck[x-100,y-100]
		nonSelectedImage[x-startIndexOfMask_x][y-array_dimensionsOfMask[3]] = nonNeckPixelValue
		npNonSelectedImage = np.asarray(nonSelectedImage)

	#Create array to return
	returnCellsAndNoncell=[]
	returnCellsAndNoncell.append(npSelectedImage)
	returnCellsAndNoncell.append(npNonSelectedImage)
	return returnCellsAndNoncell



"""Run the functions to process image here"""
#subject_imageNum.tif
#Every image with the same subject number comes from the same person.
for subject in range(1,48):#48
	for imageNum in range(1,121):#121
		print subject
		print imageNum
		try: 
			#subject = 1
			#imageNum = 3
			nameOfFile_mask ='train/' + str(subject) + "_" + str(imageNum) + "_mask.tif"
			nameOfFile_neck ='train/' + str(subject) + "_" + str(imageNum) + ".tif"
			inputArray = getDimensionsOfMask(nameOfFile_mask)
			
			selectedNeckImage = getDimensionsOfNeckImage(inputArray, nameOfFile_neck)

			"""Save Image - selected in Mask"""
			greyim = Image.new('L', (inputArray[2],inputArray[1]))
			greyim.putdata(selectedNeckImage[0].flatten())
			greyim.save('trainYes/image'+ str(subject) + "_" + str(imageNum) + ".png" )

			"""Save Image - Not selected in Mask(not neck cells)"""
			greyim = Image.new('L', (inputArray[2],inputArray[1]))
			greyim.putdata(selectedNeckImage[1].flatten())
			greyim.save('trainNO/no-image'+ str(subject) + "_" + str(imageNum) + ".png" )


		except:
			print("did not work for "+ nameOfFile_neck)
			#pass




