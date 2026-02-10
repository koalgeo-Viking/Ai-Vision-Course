import cv2
import numpy as np
import time

cap = cv2.VideoCapture(0)

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Hedef pencere boyutları
target_width = 1280
target_height = 720

# Orijinal video boyutları
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

width_ratio = original_width / target_width
height_ratio = original_height / target_height

lk_params = dict(winSize=(15, 15),
                 maxLevel=4,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def select_point(event, x, y, flags, params):
    global point, point_selected, old_points, lost_time
    if event == cv2.EVENT_LBUTTONDOWN:
        original_x = int(x * width_ratio)
        original_y = int(y * height_ratio)
        point = (original_x, original_y)
        point_selected = True
        old_points = np.array([[original_x, original_y]], dtype=np.float32)
        lost_time = time.time()

cv2.namedWindow("Frame")
cv2.setMouseCallback("Frame", select_point)

point_selected = False
point = ()
old_points = np.array([[]])
lost_time = 0

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
        h, w = 100, 100
        crop = frame[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
            frame[10:10+crop.shape[0], -10-crop.shape[1]:-10] = crop

            cv2.line(frame, (0, int(y)), (frame.shape[1], int(y)), (0, 255, 0), 1)
            cv2.line(frame, (int(x), 0), (int(x), frame.shape[0]), (0, 255, 0), 1)
            # Siyah gölge çizgiler (1 piksel aşağı ve sağa kaydırılmış)
            cv2.line(frame, (0, int(y)+2), (frame.shape[1], int(y)+ 2), (0, 0, 0), 1)
            cv2.line(frame, (int(x)+2, 0), (int(x)+2, frame.shape[0]), (0, 0, 0), 1)

            diamond_size = 20
            center = (int(x), int(y))

            base_pts = [
                (center[0], center[1] - diamond_size),
                (center[0] + diamond_size, center[1]),
                (center[0], center[1] + diamond_size),
                (center[0] - diamond_size, center[1])
            ]

            # 0 derece gölge elmas (offset +1, +1)
            shadow_pts_0 = np.array([
                (center[0] + 1, center[1] - diamond_size + 1),
                (center[0] + diamond_size + 1, center[1] + 1),
                (center[0] + 1, center[1] + diamond_size + 1),
                (center[0] - diamond_size + 1, center[1] + 1)
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [shadow_pts_0], isClosed=True, color=(0, 0, 0), thickness=1)

            # 180 derece gölge elmas (x,y) -> (-x,-y) + offset +1,+1
            rotated_180_shadow_pts = []
            for (px, py) in base_pts:
                dx = px - center[0]
                dy = py - center[1]
                rdx = -dx
                rdy = -dy
                rx = int(center[0] + rdx + 1)
                ry = int(center[1] + rdy + 1)
                rotated_180_shadow_pts.append((rx, ry))
            rotated_180_shadow_pts = np.array(rotated_180_shadow_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [rotated_180_shadow_pts], isClosed=True, color=(0, 0, 0), thickness=1)

            # 270 derece gölge elmas (x,y) -> (y, -x) + offset +1,+1
            rotated_270_shadow_pts = []
            for (px, py) in base_pts:
                dx = px - center[0]
                dy = py - center[1]
                rdx = dy
                rdy = -dx
                rx = int(center[0] + rdx + 1)
                ry = int(center[1] + rdy + 1)
                rotated_270_shadow_pts.append((rx, ry))
            rotated_270_shadow_pts = np.array(rotated_270_shadow_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [rotated_270_shadow_pts], isClosed=True, color=(0, 0, 0), thickness=1)

            # 360 derece gölge elmas (aslında 0 dereceyle aynı, ama offseti biraz değiştirerek göstereceğiz)
            shadow_pts_360 = np.array([
                (center[0] + 2, center[1] - diamond_size + 2),
                (center[0] + diamond_size + 2, center[1] + 2),
                (center[0] + 2, center[1] + diamond_size + 2),
                (center[0] - diamond_size + 2, center[1] + 2)
            ], np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [shadow_pts_360], isClosed=True, color=(0, 0, 0), thickness=1)

            # Ana elmas (yeşil)
            pts = np.array(base_pts, np.int32).reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], isClosed=True, color=(0, 255, 0), thickness=1)

        rect_start = (int(frame.shape[1] / 20), int(frame.shape[0] / 20))
        rect_end = (int(frame.shape[1] * 19 / 20), int(frame.shape[0] * 19 / 20))
        cv2.rectangle(frame, rect_start, rect_end, (0, 0, 255), 1)

        if x < rect_start[0] or x > rect_end[0] or y < rect_start[1] or y > rect_end[1]:
            if time.time() - lost_time >= 2:
                point_selected = False
                cv2.putText(frame, "Seçim İptal Edildi!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        coords_text = f"X: {int(x)}, Y: {int(y)}"
        text_size, _ = cv2.getTextSize(coords_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = frame.shape[0] - 20
        cv2.putText(frame, coords_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    if not point_selected:
        cv2.putText(frame, "!Nesne Secin!", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 1)

    if frame.shape[1] != target_width or frame.shape[0] != target_height:
        frame = cv2.resize(frame, (target_width, target_height))

    # Artı işareti ortada
    center_x = target_width // 2
    center_y = target_height // 2
    gap = 10
    length = 25
    half_length = length // 2

    cv2.line(frame, (center_x - gap - length, center_y), (center_x - gap, center_y), (0, 0, 0), 1)
    cv2.line(frame, (center_x + gap, center_y), (center_x + gap + length, center_y), (0, 0, 0), 1)
    cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + gap + half_length), (0, 0, 0), 1)

    center_x = target_width // 2 + 2
    center_y = target_height // 2 + 1

    cv2.line(frame, (center_x - gap - length, center_y), (center_x - gap, center_y), (0, 255, 0), 1)
    cv2.line(frame, (center_x + gap, center_y), (center_x + gap + length, center_y), (0, 255, 0), 1)
    cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + gap + half_length), (0, 255, 0), 1)

    # Ortalanmış köşe çizgili dikdörtgen
    rect_w = 300
    rect_h = 150
    corner_len = 20
    thickness = 1
    center_x = target_width // 2
    center_y = target_height // 2
    left = center_x - rect_w // 2
    right = center_x + rect_w // 2
    top = center_y - rect_h // 2
    bottom = center_y + rect_h // 2

    shadow_color = (0, 0, 0)
    main_color = (0, 255, 0)

    def draw_corner_lines(color, offset=0):
        cv2.line(frame, (left - offset, top - offset), (left + corner_len - offset, top - offset), color, thickness)
        cv2.line(frame, (left - offset, top - offset), (left - offset, top + corner_len - offset), color, thickness)
        cv2.line(frame, (right - corner_len + offset, top - offset), (right + offset, top - offset), color, thickness)
        cv2.line(frame, (right + offset, top - offset), (right + offset, top + corner_len - offset), color, thickness)
        cv2.line(frame, (left - offset, bottom + offset), (left + corner_len - offset, bottom + offset), color, thickness)
        cv2.line(frame, (left - offset, bottom - corner_len + offset), (left - offset, bottom + offset), color, thickness)
        cv2.line(frame, (right - corner_len + offset, bottom + offset), (right + offset, bottom + offset), color, thickness)
        cv2.line(frame, (right + offset, bottom - corner_len + offset), (right + offset, bottom + offset), color, thickness)

    draw_corner_lines(shadow_color, offset=1)
    draw_corner_lines(main_color, offset=0)

    cv2.imshow("Frame", frame)

    key = cv2.waitKey(20)
    if key == ord('q'):
        break
    elif key == ord('r'):
        point_selected = False

cap.release()
cv2.destroyAllWindows()

