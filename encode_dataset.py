import os
import pickle
import face_recognition

DATASET_FOLDER = 'Dataset'
encodings_dict = {}

for person_name in os.listdir(DATASET_FOLDER):
    person_path = os.path.join(DATASET_FOLDER, person_name)
    if os.path.isdir(person_path):
        encodings_dict[person_name] = []
        for img_name in os.listdir(person_path):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_path, img_name)
                print(f"Processing {img_path}")  # <-- Progress output
                try:
                    image = face_recognition.load_image_file(img_path)
                    encodings = face_recognition.face_encodings(image)
                    if encodings:
                        encodings_dict[person_name].append(encodings[0])
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")

with open('dataset_encodings.pickle', 'wb') as f:
    pickle.dump(encodings_dict, f)

print("Encodings saved to dataset_encodings.pickle")