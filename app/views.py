# Important imports
from app import app
from flask import request, render_template, url_for
import cv2
import numpy as np
from PIL import Image
import string
import random
import os
import torch
from ultralytics import YOLO

# Adding path to config
app.config['INITIAL_FILE_UPLOADS'] = 'app/static/uploads'
app.config['MODEL_PATH'] = 'app/static/model'

# Ensure model directory exists
os.makedirs(app.config['MODEL_PATH'], exist_ok=True)

# Initialize YOLO model
model = None

def load_model():
	global model
	if model is None:
		# Download and load YOLOv8n model
		model = YOLO('yolov8n.pt')

def detect_vehicles(image):
	global model
	if model is None:
		load_model()
	
	# Convert image to RGB if needed
	if len(image.shape) == 2:  # If grayscale
		image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
	elif image.shape[2] == 4:  # If RGBA
		image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
	
	# Run inference
	results = model(image, conf=0.3)  # Lower confidence threshold for detection
	
	# Initialize counters and detection lists
	cars = []
	buses = []
	trucks = []
	
	# Process detections
	for result in results:
		boxes = result.boxes
		for box in boxes:
			# Get box coordinates
			x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
			cls = int(box.cls[0].item())
			conf = float(box.conf[0].item())
			
			# Convert coordinates to integer
			x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
			
			# Get class name
			class_name = result.names[cls].lower()
			
			# Filter relevant vehicle classes and high confidence detections
			if conf > 0.3:  # Confidence threshold
				if class_name in ['car', 'sedan', 'suv']:
					cars.append((x1, y1, x2-x1, y2-y1))
				elif class_name in ['bus']:
					buses.append((x1, y1, x2-x1, y2-y1))
				elif class_name in ['truck']:
					trucks.append((x1, y1, x2-x1, y2-y1))
	
	return np.array(cars), np.array(buses), np.array(trucks)

# Route to home page
@app.route("/", methods=["GET", "POST"])
def index():
	if request.method == "GET":
		full_filename = 'images/white_bg.jpg'
		return render_template("index.html", full_filename=full_filename)

	if request.method == "POST":
		image_upload = request.files['image_upload']
		imagename = image_upload.filename

		# Generating unique name to save image
		letters = string.ascii_lowercase
		name = ''.join(random.choice(letters) for i in range(10)) + '.png'
		full_filename = 'uploads/' + name

		# Read and process image
		image = Image.open(image_upload)
		# Calculate new size while maintaining aspect ratio
		max_dimension = 1024
		ratio = min(max_dimension/float(image.size[0]), max_dimension/float(image.size[1]))
		new_size = tuple([int(x*ratio) for x in image.size])
		image = image.resize(new_size, Image.Resampling.LANCZOS)
		image_arr = np.array(image)

		# Detect vehicles using YOLO
		cars, buses, trucks = detect_vehicles(image_arr)

		# Draw detections
		# Cars in blue
		ccnt = 0
		if len(cars) > 0:
			for (x, y, w, h) in cars:
				cv2.rectangle(image_arr, (x,y), (x+w,y+h), (255,0,0), 2)
				ccnt += 1

		# Buses in green
		bcnt = 0
		if len(buses) > 0:
			for (x, y, w, h) in buses:
				cv2.rectangle(image_arr, (x,y), (x+w,y+h), (0,255,0), 2)
				bcnt += 1

		# Trucks in red
		tcnt = 0
		if len(trucks) > 0:
			for (x, y, w, h) in trucks:
				cv2.rectangle(image_arr, (x,y), (x+w,y+h), (0,0,255), 2)
				tcnt += 1

		# Save the processed image
		img = Image.fromarray(image_arr, 'RGB')
		img.save(os.path.join(app.config['INITIAL_FILE_UPLOADS'], name))

		# Return template with results
		result = f'{ccnt} cars, {bcnt} buses, and {tcnt} trucks detected'
		return render_template('index.html', full_filename=full_filename, pred=result)

# Main function
if __name__ == '__main__':
	app.run(debug=True)
