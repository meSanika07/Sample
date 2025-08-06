import cv2
import matplotlib.pyplot as plt
import time
import math
import datetime
import os

# Load Haar cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')

cap = cv2.VideoCapture(0)

smile_scores = []
timestamps = []
start_time = time.time()

# Optional: create folder to store steps
os.makedirs("processing_steps", exist_ok=True)

def show_processing_steps(image):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Original
    cv2.imshow("Step 1: Original Image", image)
    cv2.imwrite(f"processing_steps/step1_original_{timestamp}.png", image)
    cv2.waitKey(0)

    # 2. Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imshow("Step 2: Grayscale", gray)
    cv2.imwrite(f"processing_steps/step2_grayscale_{timestamp}.png", gray)
    cv2.waitKey(0)

    # 3. Histogram Equalization
    gray_eq = cv2.equalizeHist(gray)
    cv2.imshow("Step 3: Histogram Equalization", gray_eq)
    cv2.imwrite(f"processing_steps/step3_hist_eq_{timestamp}.png", gray_eq)
    cv2.waitKey(0)

    # 4. Gaussian Blur
    gray_blur = cv2.GaussianBlur(gray_eq, (5, 5), 0)
    cv2.imshow("Step 4: Gaussian Blur", gray_blur)
    cv2.imwrite(f"processing_steps/step4_gaussian_blur_{timestamp}.png", gray_blur)
    cv2.waitKey(0)

    # 5. Denoising
    gray_denoised = cv2.fastNlMeansDenoising(gray_blur, None, h=30, templateWindowSize=7, searchWindowSize=21)
    cv2.imshow("Step 5: Denoised", gray_denoised)
    cv2.imwrite(f"processing_steps/step5_denoised_{timestamp}.png", gray_denoised)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (640, 480))

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    gray = cv2.fastNlMeansDenoising(gray, None, h=30, templateWindowSize=7, searchWindowSize=21)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    if len(faces) == 0:
        cv2.putText(frame, "No faces detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        roi_gray = gray[y:y + h, x:x + w]
        roi_color = frame[y:y + h, x:x + w]

        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=15)

        if len(smiles) == 0:
            cv2.putText(frame, "No smile detected", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        for (sx, sy, sw, sh) in smiles:
            smile_area = sw * sh
            face_area = w * h
            score_ratio = smile_area / face_area
            boosted_score = math.sqrt(score_ratio) * 20
            smile_score = int(min(max(boosted_score, 1), 10))

            current_time = round(time.time() - start_time, 2)
            smile_scores.append(smile_score)
            timestamps.append(current_time)

            cv2.rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 2)
            cv2.putText(frame, f"Smile Score: {smile_score}/10", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow('Smile Detector with Score', frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        show_processing_steps(frame.copy())

cap.release()
cv2.destroyAllWindows()

# Smile score plotting
if smile_scores:
    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, smile_scores, color='green', marker='o', linestyle='-')
    plt.title("How Your Smile Changed Over Time 😀", fontsize=14)
    plt.xlabel("Time (seconds)", fontsize=12)
    plt.ylabel("Smile Score (1 to 10)", fontsize=12)

    for i in range(len(smile_scores)):
        plt.text(timestamps[i], smile_scores[i] + 0.2, str(smile_scores[i]),
                 fontsize=9, ha='center', va='bottom', color='black')

    plt.xticks(fontsize=10)
    plt.yticks(range(1, 11), fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
else:
    print("No smiles were detected during the session.")
