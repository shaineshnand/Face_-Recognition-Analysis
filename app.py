import os
import numpy as np
import face_recognition
from flask import Flask, request, render_template, url_for, redirect, session
import flask
from werkzeug.utils import secure_filename
from mtcnn import MTCNN
import cv2
import pickle
import shutil

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for session
UPLOAD_FOLDER = os.path.join('static', 'uploads')
DATASET_FOLDER = 'Dataset'

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load dataset encodings at the top, before any routes
with open('dataset_encodings.pickle', 'rb') as f:
    dataset_encodings = pickle.load(f)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        session['uploaded_image'] = filename  # Store filename in session
        # Redirect to detection dashboard after upload
        return redirect(url_for('detect_dashboard'))
    return render_template('upload.html', message="Invalid file type.")

# Add this route at the top-level (outside any function)
@app.route('/dataset/<path:filename>')
def dataset_static(filename):
    return flask.send_from_directory(DATASET_FOLDER, filename)

@app.route('/set_resolution', methods=['POST'])
def set_resolution():
    resolution = request.form.get('resolution', '224x224')
    session['resolution'] = resolution
    session['resolution_message'] = f"Resolution set to {resolution}"
    print(f"Resolution set to {resolution}")  # This will print to the terminal
    return redirect(url_for('detect_dashboard'))

def get_resolution_tuple():
    res_str = session.get('resolution', '224x224')
    try:
        w, h = map(int, res_str.split('x'))
        return (w, h)
    except Exception:
        return (224, 224)

def resize_image(image, resolution, light_level=None):
    # Step 1: Downscale to the selected resolution
    downscaled = cv2.resize(image, resolution, interpolation=cv2.INTER_AREA)
    
    # Step 2: Apply light drop if specified
    if light_level is not None and light_level != 100:
        # Convert to float and scale
        adjusted = downscaled.astype(float) * (light_level / 100.0)
        # Clip values to valid range and convert back to uint8
        downscaled = np.clip(adjusted, 0, 255).astype(np.uint8)
    
    # Step 3: Upscale back to display size (fixed, e.g., 224x224)
    display_size = (224, 224)  # or whatever your display size is
    upscaled = cv2.resize(downscaled, display_size, interpolation=cv2.INTER_NEAREST)
    return upscaled

def detect_and_recognize(image_path):
    print("Scanning dataset for matches...")
    detector = MTCNN()
    img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    # Resize image according to selected resolution and apply light level
    resolution = get_resolution_tuple()
    light_level = get_light_level()
    img = resize_image(img, resolution, light_level)
    faces = detector.detect_faces(img)
    if not faces:
        return "No face detected in image.", []
    x, y, width, height = faces[0]['box']
    top = max(y, 0)
    right = x + width
    bottom = y + height
    left = max(x, 0)
    face_location = [(top, right, bottom, left)]
    uploaded_encodings = face_recognition.face_encodings(img, known_face_locations=face_location)
    if not uploaded_encodings:
        return "No face encoding found in detected face.", []
    uploaded_encoding = uploaded_encodings[0]
    known_encodings = []
    known_names = []
    known_image_paths = []
    for person_name, encodings_list in dataset_encodings.items():
        print(f"Checking person: {person_name}")
        person_folder = os.path.join('Dataset', person_name)
        img_files = [img for img in os.listdir(person_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
        for idx, enc in enumerate(encodings_list):
            print(f"  Comparing with image {idx+1} of {person_name}")
            known_encodings.append(enc)
            known_names.append(person_name)
            if idx < len(img_files):
                known_image_paths.append(os.path.join(person_folder, img_files[idx]))
            else:
                known_image_paths.append(None)
    print("Finished scanning. Comparing faces...")
    matches = face_recognition.compare_faces(known_encodings, uploaded_encoding)
    distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
    matched_image_urls = []
    threshold = 0.6
    if len(distances) > 0:
        for idx, (is_match, dist) in enumerate(zip(matches, distances)):
            if is_match and dist < threshold:
                matched_img_path = known_image_paths[idx]
                if matched_img_path:
                    rel_path = os.path.relpath(matched_img_path, DATASET_FOLDER).replace("\\", "/")
                    matched_image_urls.append(url_for('dataset_static', filename=rel_path))
        if matched_image_urls:
            matched_names = set([known_names[idx] for idx, (is_match, dist) in enumerate(zip(matches, distances)) if is_match and dist < threshold])
            result = f"Matched: {', '.join(matched_names)}"
        else:
            result = "Not Recognized"
    else:
        result = "No known faces to compare."
    return result, matched_image_urls

@app.route('/detect', methods=['POST'])
def detect():
    filename = session.get('uploaded_image')
    if not filename:
        return render_template('upload.html', message="No image uploaded.")
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Prepare results for the original image
    original_result, original_matched = detect_and_recognize(upload_path)

    # Check if noisy image exists in session
    noisy_filename = session.get('noisy_image')
    if noisy_filename:
        noisy_path = os.path.join('static', 'Noisy', noisy_filename)
        noisy_result, noisy_matched = detect_and_recognize(noisy_path)
        noisy_image_url = url_for('static', filename=f'Noisy/{noisy_filename}')
    else:
        noisy_result, noisy_matched = None, None
        noisy_image_url = None

    return render_template(
        'result.html',
        original_image_url=url_for('static', filename=f'uploads/{filename}'),
        noisy_image_url=noisy_image_url,
        original_result=original_result,
        noisy_result=noisy_result,
        original_matched=original_matched,
        noisy_matched=noisy_matched,
        resolution=session.get('resolution', '224x224')
    )

@app.route('/add_noise', methods=['POST'])
def add_noise():
    filename = session.get('uploaded_image')
    if not filename:
        return render_template('upload.html', message="No image uploaded.")
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Read the image
    img = cv2.imread(upload_path)
    if img is None:
        return render_template('upload.html', message="Failed to load image.")

    # Add random shading-like noise
    noise = np.random.normal(loc=0, scale=50, size=img.shape).astype(np.int16)
    noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Create Noisy folder if it doesn't exist
    NOISY_FOLDER = os.path.join('static', 'Noisy')
    os.makedirs(NOISY_FOLDER, exist_ok=True)

    # Save the noisy image in the Noisy folder
    noisy_filename = f"noisy_{filename}"
    noisy_path = os.path.join(NOISY_FOLDER, noisy_filename)
    cv2.imwrite(noisy_path, noisy_img)

    # Store both original and noisy filenames in session
    session['original_image'] = filename
    session['noisy_image'] = noisy_filename

    # Get both original and noisy filenames from session
    original_filename = session.get('original_image')
    noisy_filename = session.get('noisy_image')

    def detect_and_recognize(image_path):
        detector = MTCNN()
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(img)
        if not faces:
            return "No face detected in image.", []
        x, y, width, height = faces[0]['box']
        top = max(y, 0)
        right = x + width
        bottom = y + height
        left = max(x, 0)
        face_location = [(top, right, bottom, left)]
        uploaded_encodings = face_recognition.face_encodings(img, known_face_locations=face_location)
        if not uploaded_encodings:
            return "No face encoding found in detected face.", []
        uploaded_encoding = uploaded_encodings[0]
        known_encodings = []
        known_names = []
        known_image_paths = []
        for person_name, encodings_list in dataset_encodings.items():
            person_folder = os.path.join('Dataset', person_name)
            img_files = [img for img in os.listdir(person_folder) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for idx, enc in enumerate(encodings_list):
                known_encodings.append(enc)
                known_names.append(person_name)
                if idx < len(img_files):
                    known_image_paths.append(os.path.join(person_folder, img_files[idx]))
                else:
                    known_image_paths.append(None)
        matches = face_recognition.compare_faces(known_encodings, uploaded_encoding)
        distances = face_recognition.face_distance(known_encodings, uploaded_encoding)
        matched_image_urls = []
        threshold = 0.6
        if len(distances) > 0:
            for idx, (is_match, dist) in enumerate(zip(matches, distances)):
                if is_match and dist < threshold:
                    matched_img_path = known_image_paths[idx]
                    if matched_img_path:
                        rel_path = os.path.relpath(matched_img_path, DATASET_FOLDER).replace("\\", "/")
                        matched_image_urls.append(url_for('dataset_static', filename=rel_path))
                if matched_image_urls:
                    matched_names = set([known_names[idx] for idx, (is_match, dist) in enumerate(zip(matches, distances)) if is_match and dist < threshold])
                    result = f"Matched: {', '.join(matched_names)}"
                else:
                    result = "Not Recognized"
        else:
            result = "No known faces to compare."
        return result, matched_image_urls

    # Prepare results for both images
    original_result, original_matched = ("No original image.", []) if not original_filename else detect_and_recognize(os.path.join(app.config['UPLOAD_FOLDER'], original_filename))
    noisy_result, noisy_matched = ("No noisy image.", []) if not noisy_filename else detect_and_recognize(os.path.join('static', 'Noisy', noisy_filename))

    return render_template(
        'result.html',
        original_image_url=url_for('static', filename=f'uploads/{original_filename}') if original_filename else None,
        noisy_image_url=url_for('static', filename=f'Noisy/{noisy_filename}') if noisy_filename else None,
        original_result=original_result,
        noisy_result=noisy_result,
        original_matched=original_matched,
        noisy_matched=noisy_matched
    )

@app.route('/detect', methods=['GET'])
def detect_dashboard():
    filename = session.get('uploaded_image')
    if not filename:
        return redirect(url_for('index'))
    image_url = url_for('static', filename=f'uploads/{filename}')
    # Detection status
    normal_status = session.get('normal_detection_status', 'Not done')
    noisy_status = session.get('noisy_detection_status', 'Not done')
    noise_type = session.get('noise_type', 'Gaussian')
    compare_ready = normal_status == 'Done' and noisy_status == 'Done'
    resolution = session.get('resolution', '224x224')
    resolution_message = session.pop('resolution_message', None)
    light_level = session.get('light_level', '100')
    light_level_message = session.pop('light_level_message', None)
    return render_template(
        'detect.html',
        image_url=image_url,
        normal_status=normal_status,
        noisy_status=noisy_status,
        noise_type=noise_type,
        compare_ready=compare_ready,
        resolution=resolution,
        resolution_message=resolution_message,
        light_level=light_level,
        light_level_message=light_level_message
    )

@app.route('/run_normal_detection', methods=['POST'])
def run_normal_detection():
    filename = session.get('uploaded_image')
    if not filename:
        return redirect(url_for('index'))
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Resize image before detection
    resolution = get_resolution_tuple()
    light_level = get_light_level()
    img = cv2.imread(upload_path)
    img = resize_image(img, resolution, light_level)
    temp_filename = f"resized_{filename}"
    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
    cv2.imwrite(temp_path, img)
    # Update session to use resized image for display and further processing
    session['uploaded_image'] = temp_filename
    result, matched = detect_and_recognize(temp_path)
    session['normal_detection_result'] = result
    session['normal_detection_matched'] = matched
    session['normal_detection_status'] = 'Done'
    return redirect(url_for('detect_dashboard'))

@app.route('/run_noisy_detection', methods=['POST'])
def run_noisy_detection():
    filename = session.get('uploaded_image')
    if not filename:
        return redirect(url_for('index'))
    noise_type = request.form.get('noise_type', 'Gaussian')
    session['noise_type'] = noise_type
    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Resize image before adding noise
    resolution = get_resolution_tuple()
    light_level = get_light_level()
    img = cv2.imread(upload_path)
    img = resize_image(img, resolution, light_level)
    # Read the image
    img = cv2.imread(upload_path)
    if img is None:
        session['noisy_detection_status'] = 'Failed'
        return redirect(url_for('detect_dashboard'))
    # Apply selected noise
    if noise_type == 'Salt & Pepper':
        noisy_img = img.copy()
        prob = 0.02
        thres = 1 - prob
        rnd = np.random.rand(*img.shape[:2])
        noisy_img[rnd < prob] = 0
        noisy_img[rnd > thres] = 255
    else:  # Default to Gaussian
        noise = np.random.normal(loc=0, scale=50, size=img.shape).astype(np.int16)
        noisy_img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # Save noisy image
    NOISY_FOLDER = os.path.join('static', 'Noisy')
    os.makedirs(NOISY_FOLDER, exist_ok=True)
    noisy_filename = f"noisy_{filename}"
    noisy_path = os.path.join(NOISY_FOLDER, noisy_filename)
    cv2.imwrite(noisy_path, noisy_img)
    session['noisy_image'] = noisy_filename
    # Run detection
    result, matched = detect_and_recognize(noisy_path)
    session['noisy_detection_result'] = result
    session['noisy_detection_matched'] = matched
    session['noisy_detection_status'] = 'Done'
    return redirect(url_for('detect_dashboard'))

from sklearn.metrics import confusion_matrix

@app.route('/set_light_level', methods=['POST'])
def set_light_level():
    light_level = request.form.get('light_level', '100')
    session['light_level'] = light_level
    session['light_level_message'] = f"Light level set to {light_level}%"
    print(f"Light level set to {light_level}%")  # This will print to the terminal
    return redirect(url_for('detect_dashboard'))

def get_light_level():
    light_level = session.get('light_level', '100')
    try:
        return int(light_level)
    except Exception:
        return 100

@app.route('/compare', methods=['GET'])
def compare():
    filename = session.get('uploaded_image')
    noisy_filename = session.get('noisy_image')
    if not filename or not noisy_filename:
        return redirect(url_for('detect_dashboard'))
    original_image_url = url_for('static', filename=f'uploads/{filename}')
    noisy_image_url = url_for('static', filename=f'Noisy/{noisy_filename}')
    normal_result = session.get('normal_detection_result', 'Not done')
    noisy_result = session.get('noisy_detection_result', 'Not done')
    normal_matched = session.get('normal_detection_matched', [])
    noisy_matched = session.get('noisy_detection_matched', [])
    resolution = session.get('resolution', '224x224')

    import pandas as pd
    gt_df = pd.read_csv('ground_truth_labels.csv')
    uploaded_image_name = filename.replace("resized_", "").replace("noisy_", "")
    gt_row = gt_df[gt_df['image_path'].str.contains(uploaded_image_name)]
    if not gt_row.empty:
        ground_truth_label = gt_row.iloc[0]['label']
    else:
        ground_truth_label = "Unknown"

    def get_names_from_matched(matched_urls):
        names = []
        for url in matched_urls:
            parts = url.split('/')
            if len(parts) >= 3:
                names.append(parts[-2])
        return names

    normal_pred_names = get_names_from_matched(normal_matched)
    noisy_pred_names = get_names_from_matched(noisy_matched)

    # Get all ground truth image paths for the person
    ground_truth_image_urls = []
    if ground_truth_label != "Unknown":
        # Use the CSV to get only the images for this label
        gt_images = gt_df[gt_df['label'] == ground_truth_label]['image_path'].tolist()
        for img_path in gt_images:
            rel_path = img_path.replace("\\", "/")
            ground_truth_image_urls.append(url_for('dataset_static', filename=rel_path))

    def get_tp_images(gt_urls, matched_urls):
        matched_set = set(matched_urls)
        return [url for url in gt_urls if url in matched_set]

    def get_missed_images(gt_urls, matched_urls):
        matched_set = set(matched_urls)
        return [url for url in gt_urls if url not in matched_set]

    normal_tp_images = get_tp_images(ground_truth_image_urls, normal_matched)
    noisy_tp_images = get_tp_images(ground_truth_image_urls, noisy_matched)
    normal_missed = get_missed_images(ground_truth_image_urls, normal_matched)
    noisy_missed = get_missed_images(ground_truth_image_urls, noisy_matched)

    ground_truth_count = len(ground_truth_image_urls)

    def get_confusion(gt, pred_names, gt_count):
        TP = sum(1 for name in pred_names if name == gt)
        FP = sum(1 for name in pred_names if name != gt)
        FN = max(gt_count - TP, 0) if gt_count else 0
        TN = 0  # Always zero in this setup
        return TP, FP, FN, TN

    normal_TP, normal_FP, normal_FN, normal_TN = get_confusion(ground_truth_label, normal_pred_names, ground_truth_count)
    noisy_TP, noisy_FP, noisy_FN, noisy_TN = get_confusion(ground_truth_label, noisy_pred_names, ground_truth_count)

    confusion_labels = ["TP", "FP", "FN"]
    normal_confusion_matrix = [
        [normal_TP],
        [normal_FP],
        [normal_FN]
    ]
    noisy_confusion_matrix = [
        [noisy_TP],
        [noisy_FP],
        [noisy_FN]
    ]

    normal_confusion = list(zip(confusion_labels, normal_confusion_matrix))
    noisy_confusion = list(zip(confusion_labels, noisy_confusion_matrix))

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    normal_precision = safe_div(normal_TP, (normal_TP + normal_FP))
    normal_recall = safe_div(normal_TP, (normal_TP + normal_FN))
    normal_f1 = safe_div(2 * normal_precision * normal_recall, (normal_precision + normal_recall)) if (normal_precision + normal_recall) > 0 else 0.0
    normal_accuracy = safe_div((normal_TP + normal_TN), (normal_TP + normal_TN + normal_FP + normal_FN))

    noisy_precision = safe_div(noisy_TP, (noisy_TP + noisy_FP))
    noisy_recall = safe_div(noisy_TP, (noisy_TP + noisy_FN))
    noisy_f1 = safe_div(2 * noisy_precision * noisy_recall, (noisy_precision + noisy_recall)) if (noisy_precision + noisy_recall) > 0 else 0.0
    noisy_accuracy = safe_div((noisy_TP + noisy_TN), (noisy_TP + noisy_TN + noisy_FP + noisy_FN))

    # --- ROC Curve Calculation based on detection metrics ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc

    def detection_roc(gt_df, ground_truth_label, matched_urls):
        y_true = []
        y_pred = []
        for idx, row in gt_df.iterrows():
            is_positive = (row['label'] == ground_truth_label)
            y_true.append(1 if is_positive else 0)
            img_url = url_for('dataset_static', filename=row['image_path'].replace("\\", "/"))
            detected = img_url in matched_urls
            y_pred.append(1 if detected else 0)
        return y_true, y_pred

    # Normal detection ROC
    y_true_normal, y_pred_normal = detection_roc(gt_df, ground_truth_label, normal_matched)
    # Noisy detection ROC
    y_true_noisy, y_pred_noisy = detection_roc(gt_df, ground_truth_label, noisy_matched)

    roc_curve_url_normal = None
    roc_curve_url_noisy = None

    # Always define normal_roc and noisy_roc as dicts with .fpr and .tpr
    normal_roc = {"fpr": [], "tpr": []}
    noisy_roc = {"fpr": [], "tpr": []}

    if y_true_normal and y_pred_normal:
        fpr, tpr, thresholds = roc_curve(y_true_normal, y_pred_normal)
        normal_roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, marker='o', color='blue', label=f'Normal Detection (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # Annotate thresholds
        for i, thr in enumerate(thresholds):
            plt.annotate(f'{thr:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Normal Detection)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join('static', 'roc_curve_normal.png')
        plt.savefig(roc_path)
        plt.close()
        roc_curve_url_normal = url_for('static', filename='roc_curve_normal.png')

    if y_true_noisy and y_pred_noisy:
        fpr, tpr, thresholds = roc_curve(y_true_noisy, y_pred_noisy)
        noisy_roc = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(7, 6))
        plt.plot(fpr, tpr, marker='o', color='red', label=f'Noisy Detection (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        # Annotate thresholds
        for i, thr in enumerate(thresholds):
            plt.annotate(f'{thr:.2f}', (fpr[i], tpr[i]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=8)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Noisy Detection)')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = os.path.join('static', 'roc_curve_noisy.png')
        plt.savefig(roc_path)
        plt.close()
        roc_curve_url_noisy = url_for('static', filename='roc_curve_noisy.png')

    return render_template(
        'compare.html',
        original_image_url=original_image_url,
        noisy_image_url=noisy_image_url,
        normal_result=normal_result,
        noisy_result=noisy_result,
        normal_matched=normal_matched,
        noisy_matched=noisy_matched,
        resolution=resolution,
        ground_truth_label=ground_truth_label,
        ground_truth_image_urls=ground_truth_image_urls,
        normal_tp_images=normal_tp_images,
        noisy_tp_images=noisy_tp_images,
        normal_missed=normal_missed,
        noisy_missed=noisy_missed,
        normal_TP=normal_TP,
        normal_FP=normal_FP,
        normal_FN=normal_FN,
        normal_TN=normal_TN,
        noisy_TP=noisy_TP,
        noisy_FP=noisy_FP,
        noisy_FN=noisy_FN,
        noisy_TN=noisy_TN,
        normal_confusion=normal_confusion,
        noisy_confusion=noisy_confusion,
        normal_precision=normal_precision,
        normal_recall=normal_recall,
        normal_f1=normal_f1,
        normal_accuracy=normal_accuracy,
        noisy_precision=noisy_precision,
        noisy_recall=noisy_recall,
        noisy_f1=noisy_f1,
        noisy_accuracy=noisy_accuracy,
        roc_curve_url_normal=roc_curve_url_normal,
        roc_curve_url_noisy=roc_curve_url_noisy,
        normal_roc=normal_roc,
        noisy_roc=noisy_roc
    )

@app.route('/reset', methods=['POST'])
def reset():
    session.clear()
    return redirect(url_for('index'))

