import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Target window dimensions
target_width = 1280
target_height = 720

# Original video dimensions
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Scaling ratios
width_ratio = original_width / target_width
height_ratio = original_height / target_height

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_point(event, x, y, flags, params):
    global point, point_selected, old_points, lost_time
    if event == cv2.EVENT_LBUTTONDOWN:
        # Calculate original coordinates based on the screen size
        original_x = int(x * width_ratio)
        original_y = int(y * height_ratio)
        
        point = (original_x, original_y)
        point_selected = True
        old_points = np.array([[original_x, original_y]], dtype=np.float32)
        lost_time = time.time()  # Reset the timer when a new point is selected

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([[]])
lost_time = 0  # The time when the object was lost

while True:
    _, frame = cap.read()
    if frame is None:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if point_selected:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        x, y = new_points[0][0], new_points[0][1]
        h, w = 100, 100  # Smaller zoom window size
        crop = frame[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
            frame[10:10+crop.shape[0], -10-crop.shape[1]:-10] = crop

            # Draw horizontal and vertical lines (with thinner thickness)
            cv2.line(frame, (0, int(y)), (frame.shape[1], int(y)), (0, 255, 0), 1)  # Horizontal line
            cv2.line(frame, (int(x), 0), (int(x), frame.shape[0]), (0, 255, 0), 1)  # Vertical line

            # Draw a 20x20 green square at the intersection point of the lines
            square_size = 20
            square_x = int(x) - square_size // 2
            square_y = int(y) - square_size // 2
            cv2.rectangle(frame, (square_x, square_y), (square_x + square_size, square_y + square_size), (0, 255, 0), 2)

        cv2.circle(frame, (int(x), int(y)), 5, (0, 0, 255), -1)  # Draw a red circle to indicate the tracking point

        # Coordinates for the rectangle (adjusted to be 6/4 of the window size)
        rect_start = (int(frame.shape[1] / 20), int(frame.shape[0] / 20))  # Start coordinates
        rect_end = (int(frame.shape[1] * 19 / 20), int(frame.shape[0] * 19 / 20))  # End coordinates

        # Draw the rectangle
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 1)

        # If the object moves outside the rectangle and a certain amount of time has passed, cancel the selection
        if x < rect_start[0] or x > rect_end[0] or y < rect_start[1] or y > rect_end[1]:
            if time.time() - lost_time >= 2:  # Cancel after waiting 2 seconds
                point_selected = False
                cv2.putText(frame, "Selection Cancelled!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the coordinates of the object in the bottom-right corner of the screen
        coords_text = f"X: {int(x)}, Y: {int(y)}"
        text_size, _ = cv2.getTextSize(coords_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = frame.shape[0] - 20
        cv2.putText(frame, coords_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if not point_selected:
        cv2.putText(frame, "!Select an Object!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    # Resize the frame if its dimensions differ from the target
    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(frame, (target_width, target_height))

    cv2.imshow("Frame", frame)

    # Wait 40 ms between each frame to slow down the video
    key = cv2.waitKey(40)
    if key == ord('q'):
        break
    elif key == ord('r'):  # Press 'r' to cancel selection
        point_selected = False

cap.release()
cv2.destroyAllWindows()
