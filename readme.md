# Vehicle Detection and Counting System

A Flask-based web application that detects and counts vehicles in images using YOLOv8 AI model.

## Features

- Upload and process images containing vehicles
- Detect multiple types of vehicles (cars, buses, trucks)
- Real-time counting of detected vehicles
- Modern and responsive web interface
- AI-powered vehicle detection using YOLOv8
- Color-coded detection boxes for different vehicle types

## Technologies Used

- Python 3.8+
- Flask
- OpenCV
- PyTorch
- YOLOv8
- HTML/CSS
- Materialize CSS

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Razamirxa/Detect-and-Count-Vehicle-Flask-App.git
cd Detect-and-Count-Vehicle-Flask-App
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Flask application:
```bash
python app.py
```

2. Open your web browser and navigate to `http://127.0.0.1:5000`

3. Upload an image containing vehicles

4. The system will process the image and display:
   - Detected vehicles with colored bounding boxes
   - Count of different vehicle types
   - Cars: Blue boxes
   - Buses: Green boxes
   - Trucks: Red boxes

## Project Structure

```
├── app/
│   ├── static/
│   │   ├── css/
│   │   ├── js/
│   │   ├── uploads/
│   │   └── model/
│   ├── templates/
│   │   └── index.html
│   ├── __init__.py
│   └── views.py
├── config.py
├── requirements.txt
└── app.py
```

## Contributing

Feel free to fork the project and submit pull requests for any improvements.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
