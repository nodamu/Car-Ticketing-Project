import sys, os
import keras
import cv2
import traceback

from utils.keras_utils import load_model
from glob import glob
from os.path import splitext, basename
from utils.utils import im2single
from utils.keras_utils import load_model, detect_lp
from utils.label import Shape, writeShapes


def adjust_pts(pts,lroi):
	return pts*lroi.wh().reshape((2,1)) + lroi.tl().reshape((2,1))


if __name__ == '__main__':

	img_dir = "samples/train-detector/new1.jpg"
	wpod_net_path = "data/lp-detector/wpod-net_update1.h5"
	model = load_model(wpod_net_path)
	vehicle = cv2.imread(img_dir)
	ratio = float(max(vehicle.shape[:2]))/min(vehicle.shape[:2])
	side  = int(ratio*288.)
	bound_dim = min(side + (side%(2**4)),608)
	lp_threshold = .5
	print("\t\tBound dim: %d, ratio: %f" % (bound_dim,ratio))

	Llp,LlpImgs,_ = detect_lp(model,im2single(vehicle),bound_dim,2**4,(240,80),lp_threshold)

	if len(LlpImgs):
		Ilp = LlpImgs[0]
		Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
		Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

		s = Shape(Llp[0].pts)

		# cv2.imwrite('test_lp.png' ,Ilp*255.)
		# writeShapes('test_lp.txt',[s])


	

