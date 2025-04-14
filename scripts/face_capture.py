import cv2
import os

name = input("Enter the name of the person: ")
dataset_dir = f"../dataset/{name}"
os.makedirs(dataset_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print("[INFO] Press SPACE to capture image, ESC to quit, or Q to close camera.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Capture Face", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == 27:  # ESC
        print("[INFO] Exiting...")
        break
    elif key == ord(' '):  # SPACE
        img_path = os.path.join(dataset_dir, f"{name}_{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"[INFO] Image saved: {img_path}")
        count += 1
    elif key == ord('q') or key == ord('Q'):  # Q
        print("[INFO] Camera closed by user (Q pressed).")
        break

cap.release()
cv2.destroyAllWindows()