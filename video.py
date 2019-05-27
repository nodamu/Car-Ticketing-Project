import imutils 
import cv2
import pytesseract
import os
import numpy as np
from imutils.object_detection import non_max_suppression
import argparse
from utils.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from utils.utils import im2single
from utils.keras_utils import load_model, detect_lp
from utils.label import Shape, writeShapes


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,
	help="path to input image")
# ap.add_argument("-east", "--east", type=str,
# 	help="path to input EAST text detector")
ap.add_argument("-c", "--min-confidence", type=float, default=0.6,
	help="minimum probability required to inspect a region")
ap.add_argument("-w", "--width", type=int, default=160,
	help="nearest multiple of 32 for resized width")
ap.add_argument("-e", "--height", type=int, default=128,
	help="nearest multiple of 32 for resized height")
ap.add_argument("-p", "--padding", type=float, default=0.0,
	help="amount of padding to add to each border of ROI")
args = vars(ap.parse_args())


def decode_predictions(scores, geometry):
	# grab the number of rows and columns from the scores volume, then
	# initialize our set of bounding box rectangles and corresponding
	# confidence scores
	(numRows, numCols) = scores.shape[2:4]
	rects = []
	confidences = []

	# loop over the number of rows
	for y in range(0, numRows):
		# extract the scores (probabilities), followed by the
		# geometrical data used to derive potential bounding box
		# coordinates that surround text
		scoresData = scores[0, 0, y]
		xData0 = geometry[0, 0, y]
		xData1 = geometry[0, 1, y]
		xData2 = geometry[0, 2, y]
		xData3 = geometry[0, 3, y]
		anglesData = geometry[0, 4, y]

		# loop over the number of columns
		for x in range(0, numCols):
			# if our score does not have sufficient probability,
			# ignore it
			if scoresData[x] < args["min_confidence"]:
				continue

			# compute the offset factor as our resulting feature
			# maps will be 4x smaller than the input image
			(offsetX, offsetY) = (x * 4.0, y * 4.0)

			# extract the rotation angle for the prediction and
			# then compute the sin and cosine
			angle = anglesData[x]
			cos = np.cos(angle)
			sin = np.sin(angle)

			# use the geometry volume to derive the width and height
			# of the bounding box
			h = xData0[x] + xData2[x]
			w = xData1[x] + xData3[x]

			# compute both the starting and ending (x, y)-coordinates
			# for the text prediction bounding box
			endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
			endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
			startX = int(endX - w)
			startY = int(endY - h)

			# add the bounding box coordinates and probability score
			# to our respective lists
			rects.append((startX, startY, endX, endY))
			confidences.append(scoresData[x])

	# return a tuple of the bounding boxes and associated confidences
	return (rects, confidences) 


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))

#  ROI EXTRACTION SECTION
# # img_dir = "samples/train-detector/new1.jpg"
wpod_net_path = "models/wpod-net_update1.h5"
east_model_path = "models/frozen_east_text_detection.pb"
model = load_model(wpod_net_path)

# vehicle = cv2.imread(args["image"])
# ratio = float(max(vehicle.shape[:2]))/min(vehicle.shape[:2])
# side  = int(ratio*288.)
# bound_dim = min(side + (side%(2**4)),608)
# lp_threshold = .5
# print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

vs = cv2.VideoCapture('vids/VID-20190526-WA0007.mp4')
writer = None
(W,H) = (None,None)

try:
    prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT if imutils.is_cv2 \
        else cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print("[INFO] {} total frames in video".format(total))

except:
    print("[INFO] could not determine the number of frames in video")
    total = -1

while True:
    (grabbed,frame ) = vs.read()
    
    # If frame is not grabbed then we have come to end of frame
    if not grabbed:
        break

    if W is None or H is None:
        (H,W) = frame.shape[:2]
    
    
    ratio = float(max(frame.shape[:2]))/min(frame.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    lp_threshold = .5
    # print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))
    cv2.imshow("Video", frame)
    img = frame.copy
    Llp,LlpImgs,_ = detect_lp(model,im2single(),bouimgnd_dim,2**4,(240,80),lp_threshold)

    if len(LlpImgs):
        Ilp = LlpImgs[0]
        Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

        s = Shape(Llp[0].pts)
        print(Llp[0].pts)
        cv2.imshow("Text Detection",Ilp)
        # cv2.waitKey(0)
    # cv2.destroyAllWindows()
        # print(s)
        # Write cropped license plate file to output directory
        # cv2.imwrite('output/temp.png',Ilp*255.)
        # writeShapes('test_lp.txt',[s])
        
        

    # # OCR SECTION
    # # load the input image and grab the image dimensions
    # image = cv2.imread('output/temp.png')
    # orig = image.copy()
    # (origH, origW) = image.shape[:2]

    # # set the new width and height and then determine the ratio in change
    # # for both the width and height
    # (newW, newH) = (args["width"], args["height"])
    # rW = origW / float(newW)
    # rH = origH / float(newH)

    # # resize the image and grab the new image dimensions
    # image = cv2.resize(image, (newW, newH))
    # (H, W) = image.shape[:2]

    # # define the two output layer names for the EAST detector model that
    # # we are interested in -- the first is the output probabilities and the
    # # second can be used to derive the bounding box coordinates of text
    # layerNames = [
    #     "feature_fusion/Conv_7/Sigmoid",
    #     "feature_fusion/concat_3"]

    # # load the pre-trained EAST text detector
    # print("[INFO] loading EAST text detector...")
    # net = cv2.dnn.readNet(east_model_path)
    # cv2.imshow("License Plate",image)
    # cv2.waitKey(0)
    # # construct a blob from the image and then perform a forward pass of
    # # the model to obtain the two output layer sets
    # blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
    #     (123.68, 116.78, 103.94), swapRB=True, crop=False)
    # net.setInput(blob)

    # (scores, geometry) = net.forward(layerNames)

    # # decode the predictions, then  apply non-maxima suppression to
    # # suppress weak, overlapping bounding boxes
    # (rects, confidences) = decode_predictions(scores, geometry)
    # boxes = non_max_suppression(np.array(rects), probs=confidences)

    # # initialize the list of results
    # results = []

    # # loop over the bounding boxes
    # for (startX, startY, endX, endY) in boxes:
    #     # scale the bounding box coordinates based on the respective
    #     # ratios
    #     startX = int(startX * rW)
    #     startY = int(startY * rH)
    #     endX = int(endX * rW)
    #     endY = int(endY * rH)

    #     # in order to obtain a better OCR of the text we can potentially
    #     # apply a bit of padding surrounding the bounding box -- here we
    #     # are computing the deltas in both the x and y directions
    #     dX = int((endX - startX) * args["padding"])
    #     dY = int((endY - startY) * args["padding"])

    #     # apply padding to each side of the bounding box, respectively
    #     startX = max(0, startX - dX)
    #     startY = max(0, startY - dY)
    #     endX = min(origW, endX + (dX * 2))
    #     endY = min(origH, endY + (dY * 2))

    #     # extract the actual padded ROI
    #     roi = orig[startY:endY, startX:endX]
    #     # initialize the list of results
    # results = []

    # # loop over the bounding boxes
    # for (startX, startY, endX, endY) in boxes:
    #     # scale the bounding box coordinates based on the respective
    #     # ratios
    #     startX = int(startX * rW)
    #     startY = int(startY * rH)
    #     endX = int(endX * rW)
    #     endY = int(endY * rH)

    #     # in order to obtain a better OCR of the text we can potentially
    #     # apply a bit of padding surrounding the bounding box -- here we
    #     # are computing the deltas in both the x and y directions
    #     dX = int((endX - startX) * args["padding"])
    #     dY = int((endY - startY) * args["padding"])

    #     # apply padding to each side of the bounding box, respectively
    #     startX = max(0, startX - dX)
    #     startY = max(0, startY - dY)
    #     endX = min(origW, endX + (dX * 2))
    #     endY = min(origH, endY + (dY * 2))

    #     # extract the actual padded ROI
    #     roi = orig[startY:endY, startX:endX]

    #     # in order to apply Tesseract v4 to OCR text we must supply
    #     # (1) a language, (2) an OEM flag of 4, indicating that the we
    #     # wish to use the LSTM neural net model for OCR, and finally
    #     # (3) an OEM value, in this case, 7 which implies that we are
    #     # treating the ROI as a single line of text
    #     config = ("-l eng --oem 1 --psm 7")
    #     gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    #     gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #     text = pytesseract.image_to_string(gray, config=config)

    #     # add the bounding box coordinates and OCR'd text to the list
    #     # of results
    #     results.append(((startX, startY, endX, endY), text))

    # #--------------------------------------API SECTION -----------------------------------------
    # # sort the results bounding box coordinates from top to bottom
    # # contains results of OCR system
    # results = sorted(results, key=lambda r:r[0][1])

    # # Add regex to filter false postives
    # # Print only text
    # for i in results:
    # 	print(i[1])
    # # loop over the results
    # #-------------------------------------------------------------------------------------------

    # # 
    # # for ((startX, startY, endX, endY), text) in results:
    # #     # # display the text OCR'd by Tesseract
    # #     print("OCR TEXT")
    # #     print("========")
    # #     print("{}\n".format(text))
        

    # #     # strip out non-ASCII text so we can draw the text on the image
    # #     # using OpenCV, then draw the text and a bounding box surrounding
    # #     # the text region of the input image
    # #     text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    # #     output = orig.copy()
    # #     # 	img = cv2.resize(output,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_CUBIC)
    # #     cv2.rectangle(output, (startX, startY), (endX, endY),
    # #         (0, 0, 255), 2)
    # #     cv2.putText(output, text, (startX, startY - 20),
    # #         cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    # #     # show the output image
    # #     cv2.imshow("Text Detection", output)
    # #     cv2.waitKey(0)
