Face_-Recognition-Analysis
# Face Recognition System
A comprehensive face recognition system that allows users to upload images, detect faces, and compare recognition performance under normal and noisy conditions.

## Features
- Face Detection and Recognition : Utilizes MTCNN for face detection and face_recognition library for facial recognition
- Noise Analysis : Compare recognition performance between normal and noisy images
- Multiple Noise Types : Support for Gaussian and Salt & Pepper noise
- Resolution Control : Adjust image resolution for processing
- Light Level Adjustment : Modify image brightness for testing recognition under different lighting conditions
- Performance Metrics : Comprehensive metrics including:
  - Confusion matrices
  - Precision, Recall, F1 Score, and Accuracy
  - ROC curves for performance visualization
  - 
## Installation
1. Clone the repository:
   https://github.com/shaineshnand/Face_-Recognition-Analysis.git
3. Install required dependencies:
    pip install flask
    pip install numpy
    pip install face_recognition
    pip install werkzeug
    pip install mtcnn
    pip install opencv-python
    pip install scikit-learn
    pip install pandas
    pip install matplotlib
   
## Commands to Run the Application
### 1. Generate Face Encodings
Before running the application, you need to generate encodings for your dataset:
```
python encode_dataset.py
```
### 2. Generate Ground Truth Labels
This creates the ground truth labels for evaluation metrics:
```
python 
generate_ground_truth.py
```
### 3. Run the  Application
Start the web application:
```
flask run
```
The application will be available at http://localhost:5000 (or the port specified in your app.py).

## Workflow
1. Upload Image : Upload an image containing faces to be recognized
2. Detection Dashboard : Set resolution and light level, then run normal and noisy detection
3. Compare Results : View side-by-side comparison of normal vs. noisy detection with detailed metrics
## Project Structure
- app.py : Main Flask application with routes and face recognition logic
- encode_dataset.py : Script to pre-encode faces from the dataset
- templates/ : HTML templates for the web interface
  - base.html : Base template with common elements
  - detect.html : Detection dashboard interface
  - compare.html : Results comparison interface
  - result.html : Individual detection results
- static/ : Static files (CSS, uploaded images, result images)
- Dataset/ : Face image dataset organized by person
- 
## Performance Metrics
The system calculates and displays:
- True Positives (TP), False Positives (FP), False Negatives (FN)
- Precision: TP / (TP + FP)
- Recall: TP / (TP + FN)
- F1 Score: 2 × (Precision × Recall) / (Precision + Recall)
- Accuracy: (TP + TN) / (TP + TN + FP + FN)
- ROC Curves for visual performance analysis

## Acknowledgments
- Face recognition powered by face_recognition
- Face detection using MTCNN
  
## Contributors
Shainesh Nand  - S11208989
Saryu Kumar    - S11182150
Ajnesh Deo     - S11210959
Anshika Rao    - S11208306

