import os

import cv2
import numpy as np


def upload(i):
    global files, image
    image = cv2.imread(r'{}/{}'.format(path_pliki, files[i-ord('0')]))
    cv2.imshow('obrazek', image)


def create_mask():
    global image
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 130])
    upper = np.array([180, 60, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return mask


def sift3(image_ref, frame):
    k = 5
    object_mask = cv2.bitwise_not(create_background_mask(image_ref))
    object_only = cv2.bitwise_and(image_ref, image_ref, mask=object_mask)
    gimg1 = cv2.cvtColor(object_only, cv2.COLOR_BGR2GRAY)

    gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    gimg1 = cv2.medianBlur(gimg1, k)
    gimg2 = cv2.medianBlur(gimg2, k)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gimg1, None)
    kp2, des2 = sift.detectAndCompute(gimg2, None)

    if des1 is None or des2 is None:
        return 0, frame, []

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)


    matched_pts = [kp2[m.trainIdx].pt for m in good_matches[:10]]
    return len(good_matches), frame, matched_pts


def orb_1():
    global image
    object_mask = cv2.bitwise_not(create_background_mask(image))
    object_only = cv2.bitwise_and(image, image, mask=object_mask)

    gimg1 = cv2.cvtColor(object_only, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.1,
        nlevels=10,
        edgeThreshold=10,
        patchSize=49,
        fastThreshold=5
    )

    keypoint, descriptors_1 = orb.detectAndCompute(gimg1, None)
    keypointimage = cv2.drawKeypoints(image, keypoint, None, color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('obrazek', keypointimage)


def orb(frame, image_ref):
    object_mask = cv2.bitwise_not(create_background_mask(image_ref))  # maska obiektu
    object_only = cv2.bitwise_and(image_ref, image_ref, mask=object_mask)
    gimg1 = cv2.cvtColor(object_only, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(
        nfeatures=2000,
        scaleFactor=1.1,
        nlevels=10,
        edgeThreshold=10,
        patchSize=49,
        fastThreshold=5
    )
    keypoints_1, descriptors_1 = orb.detectAndCompute(gimg1, None)
    keypoints_2, descriptors_2 = orb.detectAndCompute(gimg2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descriptors_1, descriptors_2)

    matches = sorted(matches, key=lambda x: x.distance)

    matched_points = [keypoints_2[m.trainIdx].pt for m in matches[:30]]

    return len(matches), frame, matched_points


def sift1():
    global image
    object_mask = cv2.bitwise_not(create_background_mask(image))
    object_only = cv2.bitwise_and(image, image, mask=object_mask)

    gimg = cv2.cvtColor(object_only, cv2.COLOR_BGR2GRAY)

    siftobject = cv2.SIFT_create()
    keypoint, descriptor = siftobject.detectAndCompute(gimg, object_mask)

    keypointimage = cv2.drawKeypoints(image, keypoint, None, color=(0, 255, 0),
                                      flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    cv2.imshow('obrazek', keypointimage)


def sift2(frame, image_ref):
    object_mask = cv2.bitwise_not(create_background_mask(image_ref))
    object_only = cv2.bitwise_and(image_ref, image_ref, mask=object_mask)

    gimg = cv2.cvtColor(object_only, cv2.COLOR_BGR2GRAY)
    gimg2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(gimg, object_mask)
    kp2, des2 = sift.detectAndCompute(gimg2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    matched_points = [kp2[match.trainIdx].pt for match in matches[:30]]

    return len(matches), frame, matched_points


def create_background_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 0, 130])
    upper = np.array([180, 60, 255])

    mask = cv2.inRange(hsv, lower, upper)
    return mask

def change_h(x):
    global fun
    if fun is not None:
        fun()

path_pliki = r'pliki'
image = None
fun = None
files = None

def process_frame(frame, files, is_orb):
    best_match_count = 0
    best_points = []
    best_file = None

    for file in files:
        image_ref = cv2.imread(os.path.join(path_pliki, file))
        if not is_orb:
            match_count, _, points = sift3(image_ref, frame)
        else:
            match_count, _, points = orb(image_ref, frame)

        if match_count > best_match_count:
            best_match_count = match_count
            best_file = file
            best_points = points

    result = frame.copy()

    if best_points:
        points = np.array(best_points, dtype=np.int32)

        if len(points) > 0:
            points = points.reshape(-1, 1, 2)
            x, y, w, h = cv2.boundingRect(points)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return result, best_file


def process_video(input_path, output_path, is_orb):
    video = cv2.VideoCapture(input_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    size = (frame_width, frame_height)
    result = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MJPG'), 30, size)

    files = os.listdir(path_pliki)
    match_stats = {}

    counter = 1
    while True:
        success, frame = video.read()
        if not success:
            break

        print(f"Klatka {counter} z {total_frames}")
        processed_frame, best_img = process_frame(frame, files, is_orb)

        if best_img:
            match_stats[best_img] = match_stats.get(best_img, 0) + 1

        result.write(processed_frame)
        counter += 1

    video.release()
    result.release()

    print("\nStatystyki dopasowań:")
    for k, v in match_stats.items():
        print(f"{k}: {v} dopasowań")

    if match_stats:
        best = max(match_stats, key=match_stats.get)
        print(f"\nNajwięcej dopasowań miał: {best} ({match_stats[best]} dopasowań)")

def main():
    global image, fun, files

    files = os.listdir(path_pliki)
    upload(ord('0'))
    nimg = image.copy()
    cv2.createTrackbar('low', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('high', 'obrazek', 0, 255, change_h)
    cv2.createTrackbar('ksize', 'obrazek', 5, 50, change_h)

    while True:
        key = cv2.waitKey()
        if key >= ord('0') and key <= ord('9'):
            upload(key)
            orb_1()
        elif key == ord('b'):
            sift1()
            fun = sift1
        elif key == ord('n'):
            sift2()
            fun = sift2
        elif key == ord('m'):
            sift3()
            fun = sift3
        elif key == ord('l'):
            process_video('/IMG_6623.mp4',
                          'wynik.mp4', True)
        elif key == ord('p'):
            process_video('/IMG_6623.mp4',
                          'wynik.mp4', False)
        elif key == 27:
            cv2.destroyAllWindows()
            break


if __name__ == '__main__':
    main()
