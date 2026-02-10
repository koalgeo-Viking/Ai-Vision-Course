import cv2
import numpy as np
import time

cap = cv2.VideoCapture('C:/Users/alper/Desktop/7.mp4')

_, frame = cap.read()
old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

target_width = 1280
target_height = 720

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

color_toggle = True
blink_interval = 0.5
last_blink_time = 0
circle_visible = True

bw_mode = False

while True:
    _, frame = cap.read()
    if frame is None:
        break

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    main_color = (0, 255, 0) if color_toggle else (255, 255, 255)

    if point_selected:
        new_points, status, error = cv2.calcOpticalFlowPyrLK(old_gray, gray_frame, old_points, None, **lk_params)
        old_gray = gray_frame.copy()
        old_points = new_points

        x, y = new_points[0][0], new_points[0][1]
        h, w = 225, 225
        crop = frame[int(y - h/2):int(y + h/2), int(x - w/2):int(x + w/2)]

        if crop.shape[0] > 0 and crop.shape[1] > 0:
            frame[10:10+crop.shape[0], -10-crop.shape[1]:-10] = crop

            cv2.line(frame, (0, int(y)), (frame.shape[1], int(y)), main_color, 1)
            cv2.line(frame, (int(x), 0), (int(x), frame.shape[0]), main_color, 1)
            cv2.line(frame, (0, int(y)+2), (frame.shape[1], int(y)+2), (0, 0, 0), 1)
            cv2.line(frame, (int(x)+2, 0), (int(x)+2, frame.shape[0]), (0, 0, 0), 1)

            diamond_size = 20
            center = (int(x), int(y))
            base_pts = [
                (center[0], center[1] - diamond_size),
                (center[0] + diamond_size, center[1]),
                (center[0], center[1] + diamond_size),
                (center[0] - diamond_size, center[1])
            ]
            pts = np.array(base_pts, np.int32).reshape((-1, 1, 2))
            shadow_shift = 3
            shadow_pts = pts + np.array([shadow_shift, shadow_shift], dtype=np.int32)
            cv2.polylines(frame, [shadow_pts], isClosed=True, color=(0, 0, 0), thickness=3)
            cv2.polylines(frame, [pts], isClosed=True, color=main_color, thickness=1)

            rect_w = 600
            rect_h = 300
            center_x = frame.shape[1] // 2
            center_y = frame.shape[0] // 2
            rect_top_left = (center_x - rect_w // 2, center_y - rect_h // 2)
            rect_bottom_right = (center_x + rect_w // 2, center_y + rect_h // 2)

            inside_rect = (rect_top_left[0] <= x <= rect_bottom_right[0]) and (rect_top_left[1] <= y <= rect_bottom_right[1])
            current_time = time.time()

            if inside_rect:
                if current_time - last_blink_time > blink_interval:
                    circle_visible = not circle_visible
                    last_blink_time = current_time
            else:
                circle_visible = False

            if inside_rect and circle_visible:
                circle_radius = 25
                circle_center = (int(x), int(y))
                shadow_center = (int(x + shadow_shift), int(y + shadow_shift))
                cv2.circle(frame, shadow_center, circle_radius, (0, 0, 0), thickness=3)
                cv2.circle(frame, circle_center, circle_radius, main_color, thickness=1)

            if inside_rect:
                text = "TARGET AREA"
                font = cv2.FONT_HERSHEY_DUPLEX
                font_scale = 1.2
                thickness = 1
                text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
                text_x = (frame.shape[1] - text_size[0]) // 2
                text_y = frame.shape[0] - 40
                cv2.putText(frame, text, (text_x + 2, text_y + 2), font, font_scale, (0, 0, 0), thickness + 1)
                cv2.putText(frame, text, (text_x, text_y), font, font_scale, main_color, thickness)

        if x < frame.shape[1] / 20 or x > frame.shape[1] * 19 / 20 or y < frame.shape[0] / 20 or y > frame.shape[0] * 19 / 20:
            if time.time() - lost_time >= 2:
                point_selected = False
                cv2.putText(frame, "Selection Cancelled!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        coords_text = f"X: {int(x)}, Y: {int(y)}"
        text_size, _ = cv2.getTextSize(coords_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        text_x = frame.shape[1] - text_size[0] - 20
        text_y = frame.shape[0] - 20
        cv2.putText(frame, coords_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, main_color, 2)

    else:
        cv2.putText(frame, "SELECT OBJECT", (853, 41), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
        cv2.putText(frame, "SELECT OBJECT", (851, 40), cv2.FONT_HERSHEY_DUPLEX, 1, main_color, 1)

    display_frame = frame.copy()

    if bw_mode:
        mask = np.zeros_like(frame, dtype=np.uint8)
        if point_selected:
            cv2.line(mask, (0, int(y)), (frame.shape[1], int(y)), (255, 255, 255), 1)
            cv2.line(mask, (int(x), 0), (int(x), frame.shape[0]), (255, 255, 255), 1)
            diamond_size = 20
            base_pts = [
                (int(x), int(y) - diamond_size),
                (int(x) + diamond_size, int(y)),
                (int(x), int(y) + diamond_size),
                (int(x) - diamond_size, int(y))
            ]
            pts = np.array(base_pts, np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], (255, 255, 255))

            mask[10:10 + crop.shape[0], -10 - crop.shape[1]:-10] = 255

            font = cv2.FONT_HERSHEY_DUPLEX
            font_scale = 1.2
            thickness = 1
            text = "TARGET AREA"
            text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
            text_x = (frame.shape[1] - text_size[0]) // 2
            text_y = frame.shape[0] - 40
            cv2.rectangle(mask, (text_x, text_y - text_size[1]), (text_x + text_size[0], text_y + 5), (255, 255, 255), -1)

            coords_text = f"X: {int(x)}, Y: {int(y)}"
            coords_size, _ = cv2.getTextSize(coords_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            coords_x = frame.shape[1] - coords_size[0] - 20
            coords_y = frame.shape[0] - 20
            cv2.rectangle(mask, (coords_x, coords_y - coords_size[1] - 5), (coords_x + coords_size[0], coords_y + 5), (255, 255, 255), -1)

            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            display_frame = np.where(mask == 255, display_frame, gray_bgr)
        else:
            gray = cv2.cvtColor(display_frame, cv2.COLOR_BGR2GRAY)
            display_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if display_frame.shape[1] != target_width or display_frame.shape[0] != target_height:
        display_frame = cv2.resize(display_frame, (target_width, target_height))

    # Draw center indicators
    center_x = target_width // 2
    center_y = target_height // 2
    gap = 10
    length = 25
    half_length = length // 2

    cv2.line(display_frame, (center_x - gap - length, center_y), (center_x - gap, center_y), (0, 0, 0), 1)
    cv2.line(display_frame, (center_x + gap, center_y), (center_x + gap + length, center_y), (0, 0, 0), 1)
    cv2.line(display_frame, (center_x, center_y + gap), (center_x, center_y + gap + half_length), (0, 0, 0), 1)

    center_x += 2
    center_y += 1

    cv2.line(display_frame, (center_x - gap - length, center_y), (center_x - gap, center_y), main_color, 1)
    cv2.line(display_frame, (center_x + gap, center_y), (center_x + gap + length, center_y), main_color, 1)
    cv2.line(display_frame, (center_x, center_y + gap), (center_x, center_y + gap + half_length), main_color, 1)

    rect_w = int(300 * 1.5)
    rect_h = int(150 * 1.5)
    corner_len = 20
    thickness = 1
    left = center_x - rect_w // 2
    right = center_x + rect_w // 2
    top = center_y - rect_h // 2
    bottom = center_y + rect_h // 2

    def draw_corner_lines(color, offset=0):
        cv2.line(display_frame, (left - offset, top - offset), (left + corner_len - offset, top - offset), color, thickness)
        cv2.line(display_frame, (left - offset, top - offset), (left - offset, top + corner_len - offset), color, thickness)
        cv2.line(display_frame, (right - corner_len + offset, top - offset), (right + offset, top - offset), color, thickness)
        cv2.line(display_frame, (right + offset, top - offset), (right + offset, top + corner_len - offset), color, thickness)
        cv2.line(display_frame, (left - offset, bottom + offset), (left + corner_len - offset, bottom + offset), color, thickness)
        cv2.line(display_frame, (left - offset, bottom - corner_len + offset), (left - offset, bottom + offset), color, thickness)
        cv2.line(display_frame, (right - corner_len + offset, bottom + offset), (right + offset, bottom + offset), color, thickness)
        cv2.line(display_frame, (right + offset, bottom - corner_len + offset), (right + offset, bottom + offset), color, thickness)

    draw_corner_lines((0, 0, 0), offset=1)
    draw_corner_lines(main_color, offset=0)

    # Shortcut key instructions (top-left)
    shortcuts = [
        "'Q' - Quit",
        "'R' - Reset Selection",
        "'C' - Toggle Color",
        "'V' - B/W Mode",
        "Click to Select Object"
    ]
    start_x = 20
    start_y = 30
    line_spacing = 25
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    font_thickness = 1
    text_color = main_color
    shadow_color = (0, 0, 0)

    for i, line in enumerate(shortcuts):
        y = start_y + i * line_spacing
        cv2.putText(display_frame, line, (start_x + 1, y + 1), font, font_scale, shadow_color, font_thickness + 1, cv2.LINE_AA)
        cv2.putText(display_frame, line, (start_x, y), font, font_scale, text_color, font_thickness, cv2.LINE_AA)

    # Footer text
    footer_text1 = "DESIGNER AND DEVELOPER"
    footer_text2 = ">>>>>OPENHEIMER<<<<<"
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.5
    thickness = 1
    shadow_offset = 1
    left_margin = 20

    text_size1, _ = cv2.getTextSize(footer_text1, font, font_scale, thickness)
    text_x1 = left_margin
    text_y1 = display_frame.shape[0] - 50
    cv2.putText(display_frame, footer_text1, (text_x1 + shadow_offset, text_y1 + shadow_offset), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(display_frame, footer_text1, (text_x1, text_y1), font, font_scale, main_color, thickness)

    text_size2, _ = cv2.getTextSize(footer_text2, font, font_scale, thickness)
    text_x2 = left_margin
    text_y2 = display_frame.shape[0] - 25
    cv2.putText(display_frame, footer_text2, (text_x2 + shadow_offset, text_y2 + shadow_offset), font, font_scale, (0, 0, 0), thickness + 1)
    cv2.putText(display_frame, footer_text2, (text_x2, text_y2), font, font_scale, main_color, thickness)

    cv2.imshow("Frame", display_frame)

    key = cv2.waitKey(20)
    if key != -1:
        key = chr(key).lower()
        if key == 'q':
            break
        elif key == 'r':
            point_selected = False
        elif key == 'c':
            color_toggle = not color_toggle
        elif key == 'v':
            bw_mode = not bw_mode

cap.release()
cv2.destroyAllWindows()
