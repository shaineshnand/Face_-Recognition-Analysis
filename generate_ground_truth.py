import os
import csv

dataset_dir = "Dataset"
output_csv = "ground_truth_labels.csv"

ground_truth = []

incorrect_images = {"Aaron_Peirsol/hfytfgg.jpg", "Abba_Eban/Aaron_Tippin_0001.jpg"}  # Add all known incorrect images here

for person_name in os.listdir(dataset_dir):
    person_folder = os.path.join(dataset_dir, person_name)
    if os.path.isdir(person_folder):
        for img_file in os.listdir(person_folder):
            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(person_name, img_file)
                label = person_name
                if img_path.replace("\\", "/") in incorrect_images:
                    label = "Unknown"
                ground_truth.append([img_path, label])

# Write to CSV
with open(output_csv, "w", newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(["image_path", "label"])
    writer.writerows(ground_truth)

print(f"Ground truth labels saved to {output_csv}")