import cv2
import time
import ctypes
import pycuda.autoinit
import pycuda.driver as cuda
import matplotlib.pyplot as plt
import torch
import tensorrt as trt
import numpy as np
from skimage.util import img_as_float
from PIL import Image
from effdet.config.model_config import default_detection_model_configs
from effdet.anchors import Anchors, AnchorLabeler, generate_detections
from effdet.bench import _post_process, _batch_detection

def _size_tuple(size):
    if isinstance(size, int):
        return size, size
    else:
        assert len(size) == 2
        return size

def _pil_interp(method):
    if method == 'bicubic':
        return Image.BICUBIC                                                                  
    elif method == 'lanczos':
        return Image.LANCZOS
    elif method == 'hamming':
        return Image.HAMMING
    else:
        # default bilinear, do we want to allow nearest?
        return Image.BILINEAR

class ResizePad:

    def __init__(self, target_size: int, interpolation: str = 'bilinear', fill_color: tuple = (0, 0, 0)):
        self.target_size = _size_tuple(target_size)
        self.interpolation = interpolation
        self.fill_color = fill_color

    def __call__(self, img):

        '''
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
		'''

        width, height = img.size

        img_scale_y = self.target_size[0] / height
        img_scale_x = self.target_size[1] / width
        img_scale = min(img_scale_y, img_scale_x)
        scaled_h = int(height * img_scale)
        scaled_w = int(width * img_scale)

        new_img = Image.new("RGB", (self.target_size[1], self.target_size[0]), color=self.fill_color)
        interp_method = _pil_interp(self.interpolation)
        #cv2 to pil

        img = img.resize((scaled_w, scaled_h), interp_method)
        new_img.paste(img)
        open_cv_image = np.array(new_img, dtype=np.float32) 
        # Convert RGB to BGR 
        #open_cv_image = open_cv_image[:, :, ::-1].copy() 
        return open_cv_image


def preprocess_image(img_path):
	img = Image.open(img_path)
	Padding = ResizePad(512)
	x = Padding(img)
	z = (x - x.mean(axis = (0,1,2), keepdims=True)) / x.std(axis = (0,1,2), keepdims=True)
	z = np.transpose(z, (2,0,1)) #####CHW
	z = np.expand_dims(z, axis=0)
	z = np.ascontiguousarray(z, dtype=np.float32)
	return z

def post_processing(img, outputs):
	cls_outputs=[torch.from_numpy(outputs[0]), torch.from_numpy(outputs[1]), torch.from_numpy(outputs[2]), torch.from_numpy(outputs[3]), torch.from_numpy(outputs[4])]
	box_outputs=[torch.from_numpy(outputs[5]), torch.from_numpy(outputs[6]), torch.from_numpy(outputs[7]), torch.from_numpy(outputs[8]), torch.from_numpy(outputs[9])]
	width, height = img.size
	target_size = (512, 512)
	img_scale_y = target_size[0]/height
	img_scale_x = target_size[1]/width
	img_size = (width, height)
	img_size = torch.tensor(([[img_size[0], img_size[1]]]), dtype=torch.int32)
	img_scale = min(img_scale_y, img_scale_x)
	img_scale = torch.tensor(([img_scale]), dtype=torch.float32)
	class_out, box_out, indices, classes = _post_process(cls_outputs, box_outputs, num_levels=5, num_classes=6, max_detection_points=5000)
	config = default_detection_model_configs()
	anchors = Anchors.from_config(config)
	detection = _batch_detection(1, class_out, box_out, anchors.boxes, indices, classes, img_scale, img_size, max_det_per_image=10, soft_nms=True)
	return detection

def main():
	TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)

	# Deserialize the TensorRT engine
	print("Deserializing engine.")
	f = open("./TensorRT/Engine/effdet0.engine", "rb")
	runtime = trt.Runtime(TRT_LOGGER)
	engine = runtime.deserialize_cuda_engine(f.read())

	fps_temp = 0
	fps_pre_ave = 0
	fps_infer_ave = 0
	fps_post_ave = 0
	fps_total_ave = 0

	img_count = 50
	stop_img = img_count - 1

	for i in range(0, img_count):
		time_start = time.time()
		
		# Create numpy arrays for network outputs
		output_data_1 = np.zeros((1, 54, 64, 64))
		output_data_2 = np.zeros((1, 54, 32, 32))
		output_data_3 = np.zeros((1, 54, 16, 16))
		output_data_4 = np.zeros((1, 54, 8, 8))
		output_data_5 = np.zeros((1, 54, 4, 4))
		output_data_6 = np.zeros((1, 36, 64, 64))
		output_data_7 = np.zeros((1, 36, 32, 32))
		output_data_8 = np.zeros((1, 36, 16, 16))
		output_data_9 = np.zeros((1, 36, 8, 8))
		output_data_10 = np.zeros((1, 36, 4, 4))
		
		##############inputting the image#######################
		img_path = "./TestImages/" + str(i) + ".jpg"
		time_pre_start = time.time()
		input_data = preprocess_image(img_path)
		time_pre_end  = time.time()

		d_input = cuda.mem_alloc(1 * input_data.nbytes)

		context = engine.create_execution_context()
		# Allocate memory for output tensors
		d_output_1 = cuda.mem_alloc(1 * output_data_1.nbytes) ##########if wrong, replace the context by output-data shape
		d_output_2 = cuda.mem_alloc(1 * output_data_2.nbytes)
		d_output_3 = cuda.mem_alloc(1 * output_data_3.nbytes)
		d_output_4 = cuda.mem_alloc(1 * output_data_4.nbytes)
		d_output_5 = cuda.mem_alloc(1 * output_data_5.nbytes)
		d_output_6 = cuda.mem_alloc(1 * output_data_6.nbytes)
		d_output_7 = cuda.mem_alloc(1 * output_data_7.nbytes)
		d_output_8 = cuda.mem_alloc(1 * output_data_8.nbytes)
		d_output_9 = cuda.mem_alloc(1 * output_data_9.nbytes)
		d_output_10 = cuda.mem_alloc(1 * output_data_10.nbytes)

		bindings = [int(d_input), int(d_output_1), int(d_output_2), int(d_output_3), int(d_output_4), int(d_output_5), int(d_output_6), 
					int(d_output_7), int(d_output_8), int(d_output_9), int(d_output_10)]

		stream = cuda.Stream()

		time_infer_start = time.time()
		context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
		time_infer_end = time.time()

		cuda.memcpy_dtoh_async(output_data_1, d_output_1, stream)
		cuda.memcpy_dtoh_async(output_data_2, d_output_2, stream)
		cuda.memcpy_dtoh_async(output_data_3, d_output_3, stream)
		cuda.memcpy_dtoh_async(output_data_4, d_output_4, stream)
		cuda.memcpy_dtoh_async(output_data_5, d_output_5, stream)
		cuda.memcpy_dtoh_async(output_data_6, d_output_6, stream)
		cuda.memcpy_dtoh_async(output_data_7, d_output_7, stream)
		cuda.memcpy_dtoh_async(output_data_8, d_output_8, stream)
		cuda.memcpy_dtoh_async(output_data_9, d_output_9, stream)
		cuda.memcpy_dtoh_async(output_data_10, d_output_10, stream)

		stream.synchronize()

		'''
		print("Output Data 1:")
		print(output_data_1)
		print("Output Data 2:")
		print(output_data_2)
		print("Output Data 3:")
		print(output_data_3)
		print("Output Data 4:")
		print(output_data_4)
		print("Output Data 5:")
		print(output_data_5)
		print("Output Data 6:")
		print(output_data_6)
		print("Output Data 7:")
		print(output_data_7)
		print("Output Data 8:")
		print(output_data_8)
		print("Output Data 9:")
		print(output_data_9)
		print("Output Data 10:")
		print(output_data_10)
		'''

		###########putting all the data in the list for detections##############
		outputs = [output_data_1, output_data_2, output_data_3, output_data_4, output_data_5, output_data_6, output_data_7, output_data_8, output_data_9, output_data_10]

		#############post processing ###############
		img_detect = Image.open(img_path)
		time_post_start = time.time()
		detections = post_processing(img_detect, outputs)
		time_post_end = time.time()		

		detection = detections.tolist()
	
		##############drawing the predictions on the image############################
		img = cv2.imread(img_path)
		cv2.rectangle(img,(int(detection[0][4][0]), int(detection[0][4][1])), (int(detection[0][4][2]), int(detection[0][4][3])), (0,255,0),2)
		#img_out_path = "./TestImages_Out/" + str(i) + ".jpg"
		#cv2.imwrite(img_out_path, img)

		time_end = time.time()

		# FPS Measurments
		print("Image #{0}: ".format(i))

		latency = time_pre_end - time_pre_start
		fps_temp = 1 / latency
		print("\tPre-Processing FPS: {:.2f} FPS".format(fps_temp))
		fps_pre_ave += fps_temp

		latency = time_infer_end - time_infer_start
		fps_temp = 1 / latency
		print("\tInference FPS: {:.2f} FPS".format(fps_temp))
		fps_infer_ave += fps_temp

		latency = time_post_end - time_post_start
		fps_temp = 1 / latency
		print("\tPost-Processing FPS: {:.2f} FPS".format(fps_temp))
		fps_post_ave += fps_temp

		latency = time_end - time_start
		fps_temp = 1 / latency
		print("\tOverall FPS: {:.2f} FPS".format(fps_temp))
		fps_total_ave += fps_temp
		
		if i == stop_img:
			fps_pre_ave /= stop_img
			fps_infer_ave /= stop_img
			fps_post_ave /= stop_img
			fps_total_ave /= stop_img
			print("Average FPS Measurements:")
			print("\tAverage Pre-Processing FPS: {:.2f} FPS".format(fps_pre_ave))
			print("\tAverage Inference FPS: {:.2f} FPS".format(fps_infer_ave))
			print("\tAverage Post-Processing FPS: {:.2f} FPS".format(fps_post_ave))
			print("\tAverage Overall FPS: {:.2f} FPS".format(fps_total_ave))
			break

if __name__ == "__main__":
	main()






