import cv2
import numpy as np
import pyrealsense2 as rs
import math
import pyautogui

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

profile = pipeline.start(config)

depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

clipping_distance_in_meters = 0.5  # 1 meter
clipping_distance = clipping_distance_in_meters / depth_scale

align_to = rs.stream.color
align = rs.align(align_to)

def main():
    frame_num = 0

    while (True):
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()

        if not aligned_depth_frame or not aligned_color_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(aligned_color_frame.get_data())

        grey_color = 0
        depth_image_3d = np.dstack(
            (depth_image, depth_image, depth_image))
        color_bg_removed = np.where((depth_image_3d > clipping_distance) | (depth_image_3d <= 0), grey_color,
                                    color_image)
        img = color_bg_removed
        img = cv2.flip(img, 1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        median = cv2.GaussianBlur(gray, (5, 5), 0)
        kernel_square = np.ones((3, 3), np.uint8)
        # dilation = cv2.dilate(median, kernel_square, iterations=2)
        # opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel_square)
        opening = cv2.erode(median, kernel_square, iterations=2)
        opening = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel_square)
        ret, thresh = cv2.threshold(opening, 1, 255, cv2.THRESH_BINARY)
        contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

        if len(contours) > 0:
            contour = max(contours,key=cv2.contourArea)
            if cv2.contourArea(contour) > 2500:
                x, y, w, h = cv2.boundingRect(contour)
                thresh = thresh[y:y+h, x:x+w]
                hull = cv2.convexHull(contour, returnPoints=False)
                detects = cv2.convexityDefects(contour, hull)
                toe = 0
                for j in range(detects.shape[0]):
                    s, e, f, d = detects[j, 0]
                    start = tuple(contour[s][0])
                    end = tuple(contour[e][0])
                    far = tuple(contour[f][0])
                    a = GetDistance(start,end)
                    b = GetDistance(start,far)
                    c = GetDistance(end,far)
                    angle = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c))
                    if angle <= math.pi/2:
                        toe = toe + 1
                print(toe+1)
                cv2.putText(opening,str(toe+1),(200,200),cv2.FONT_HERSHEY_SIMPLEX,2,(255,255,255))
                frame_num = frame_num + 1
                if frame_num == 5:
                    # GestureControl(toe)
                    frame_num = 0
                cv2.imshow('gray', thresh)

        cv2.imshow('img', opening)
        key = cv2.waitKey(1)
        if key == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break


def GetDistance(beg,end):
    return (((beg[0] - end[0]) ** 2) + ((beg[1] - end[1]) ** 2)) ** 0.5


def GestureControl(num):
    if num == 4:
        pyautogui.hotkey('ctrl', '=')
    if num == 1:
        pyautogui.hotkey('ctrl', '-')
    if num == 2:
        pyautogui.scroll(80)
    if num == 3:
        pyautogui.scroll(-80)


if __name__ == "__main__":
    main()