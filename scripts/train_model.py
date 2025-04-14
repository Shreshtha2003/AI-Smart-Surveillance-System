import os
import cv2
import face_recognition
import pickle

dataset_path = "../dataset"
encoding_file = "../encodings/encodings.pkl"

known_encodings = []
known_names = []

print("[INFO] Processing dataset...")

for person_name in os.listdir(dataset_path):
    person_folder = os.path.join(dataset_path, person_name)

    if not os.path.isdir(person_folder):
        continue  # skip .DS_Store or non-folder files

    print(f"[INFO] Encoding images for: {person_name}")
    for image_name in os.listdir(person_folder):
        image_path = os.path.join(person_folder, image_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Skipping unreadable image: {image_path}")
            continue

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        boxes = face_recognition.face_locations(rgb_image)

        encodings = face_recognition.face_encodings(rgb_image, boxes)
        for encoding in encodings:
            known_encodings.append(encoding)
            known_names.append(person_name)

# Save the encodings
print("[INFO] Saving trained encodings...")
os.makedirs(os.path.dirname(encoding_file), exist_ok=True)

with open(encoding_file, "wb") as f:
    data = {"encodings": known_encodings, "names": known_names}
    pickle.dump(data, f)

print("[INFO] Training completed and saved successfully!")