import cv2
import numpy as np 

cap = cv2.VideoCapture('/media/sf_shared_folder/thresh.mp4')

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

start_time = 5.5
end_time = 13.75

start_frame = int(start_time * fps)
end_frame = int(end_time * fps)

x1, y1 = 975, 420
x2, y2 = 1375, 825

cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video_recortado_roi.mp4', fourcc, fps, (x2 - x1, y2 - y1))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    current_frame = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    
    if current_frame > end_frame:
        break
    
    if current_frame >= start_frame:
        roi = frame[y1:y2, x1:x2]
        
        roi = cv2.bilateralFilter(roi, 3, 10, 0.1)
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced = cv2.filter2D(roi, -1, kernel)

        out.write(roi)

cap.release()
out.release()
cv2.destroyAllWindows()
