#preprocessing
from PIL import Image # Python Imaging Library (PIL)
import numpy as np
np.set_printoptions(threshold='nan')#otherwise prints the truncated numpy array [0 0 0 ..., 0 0 0]


"""limit size"""
maxWidth = 50
maxHeight = 50

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

			if imarray.shape[0]<maxWidth or imarray.shape[1]<maxHeight:
				print("img"+str(subject) + "_" + str(imageNum)+"is bigger than 120x120")
			else:
				try:
					if imarray.shape[1]>maxWidth:
						#needtoDel= imarray.shape[1]-maxWidth
						#print addWidthAfter

						startIndex = (imarray.shape[1]//2) - (maxWidth//2)
						
						for j in range(0, len(imlist)):
							imlist[j] = imlist[j][startIndex:startIndex+maxWidth]

					if imarray.shape[0]>maxHeight:
						startIndex = (imarray.shape[1]//2) - (maxWidth//2)
						imlist = imlist[startIndex:startIndex+maxHeight]



					"""Save Image"""
					
					greyim = Image.new('L', (maxWidth,maxHeight))
					# L: (8-bit pixels, black and white)
					npIm =  np.asarray(imlist)
					greyim.putdata(npIm.flatten())
					greyim.save('trainNO50/no-50image'+ str(subject) + "_" + str(imageNum) + ".png" )
					
				except:
					pass
					print("img"+str(subject) + "_" + str(imageNum)+"has no mask")
		except:
			
			print("not working")
			pass





