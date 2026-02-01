import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
THRESHOLD = 30      # intensity threshold (from C code)
FRAME_GAP = 2       # n1 vs n3 comparison
# ----------------------------------------

video_path = "input_video.mp4"

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("ERROR: Cannot open video")
    exit()

frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# ---------- READ FIRST FRAME ----------
ret, first_frame = cap.read()
if not ret:
    print("ERROR: Cannot read video")
    exit()

# ---------- SELECT ROI ----------
roi = cv2.selectROI(
    "Select Object (Drag mouse, press ENTER)",
    first_frame,
    fromCenter=False,
    showCrosshair=True
)
cv2.destroyAllWindows()

x, y, w, h = map(int, roi)

# ---------- READ ALL FRAMES ----------
gray_frames = []
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frames.append(gray)

cap.release()

# ---------- CALCULATE OCCLUSION ----------
occlusion_results = []

for i in range(len(gray_frames)):
    if i < FRAME_GAP:
        occlusion_results.append(0.0)
        continue

    curr = gray_frames[i][y:y+h, x:x+w]
    past = gray_frames[i - FRAME_GAP][y:y+h, x:x+w]

    diff = cv2.absdiff(curr, past)
    disturbed_pixels = np.sum(diff > THRESHOLD)
    total_pixels = curr.size

    occlusion = (disturbed_pixels / total_pixels) * 100
    occlusion_results.append(occlusion)

# ---------- VIDEO + SCROLL BAR ----------
cap = cv2.VideoCapture(video_path)

def on_trackbar(pos):
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)

cv2.namedWindow("Video Analysis")
cv2.createTrackbar("Frame", "Video Analysis", 0, frame_count - 1, on_trackbar)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    current_frame = min(current_frame, len(occlusion_results) - 1)

    occl = occlusion_results[current_frame]
    time_sec = current_frame / fps

    display = frame.copy()
    cv2.rectangle(display, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.putText(display,
                f"Frame: {current_frame}",
                (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(display,
                f"Time: {time_sec:.2f} sec",
                (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2)

    cv2.putText(display,
                f"Occlusion: {occl:.2f} %",
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2)

    cv2.imshow("Video Analysis", display)

    if cv2.waitKey(30) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()

# ---------- FINAL GRAPH (ALWAYS WORKS) ----------
plt.figure(figsize=(10, 4))
plt.plot(occlusion_results, color="red")
plt.xlabel("Frame Number")
plt.ylabel("Occlusion Percentage")
plt.title("Object Occlusion Over Time")
plt.grid(True)

plt.savefig("occlusion_graph.png")  # GUARANTEED OUTPUT
plt.show()

# ---------- FINAL CONSOLE OUTPUT ----------
print("\n--- FINAL RESULTS ---")
print(f"Total Frames: {len(occlusion_results)}")
print(f"Average Occlusion: {np.mean(occlusion_results):.2f} %")
print("Graph saved as: occlusion_graph.png")
