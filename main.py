import os
import cv2
import imutils
import numpy as np
from solver import *
from tensorflow.keras.models import load_model

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sudoku-solver-python"))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'model-OCR.h5')

classes = np.arange(0, 10)
model = load_model(MODEL_PATH)
input_size = 48

def perspective_of_image(img, location, height=900, width=900):
    pts1 = np.float32([location[0], location[3], location[1], location[2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (width, height))

def locate_board(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bfilter = cv2.bilateralFilter(gray, 13, 20, 20)
    edged = cv2.Canny(bfilter, 30, 180)
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:15]
    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break
    return perspective_of_image(img, location), location

def render_numbers(img, numbers, color=(0, 255, 0)):
    W = int(img.shape[1] / 9)
    H = int(img.shape[0] / 9)
    for i in range(9):
        for j in range(9):
            if numbers[(j * 9) + i] != 0:
                cv2.putText(img, str(numbers[(j * 9) + i]), (i * W + int(W / 2) - int(W / 4), int((j + 0.7) * H)),
                            cv2.FONT_HERSHEY_COMPLEX, 2, color, 2, cv2.LINE_AA)
    return img

def inverse_perspective(img, masked_num, location, height=900, width=900):
    pts1 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
    pts2 = np.float32([location[0], location[3], location[1], location[2]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(masked_num, matrix, (img.shape[1], img.shape[0]))

def resize_image(img, width=800):
    aspect_ratio = img.shape[1] / img.shape[0]
    new_height = int(width / aspect_ratio)
    resized_img = cv2.resize(img, (width, new_height))
    return resized_img

def split_into_boxes(board):
    rows = np.vsplit(board, 9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r, 9)
        for box in cols:
            box = cv2.resize(box, (input_size, input_size)) / 255.0
            boxes.append(box)
    return boxes

IMAGES_DIR = os.path.join(BASE_DIR, 'image')
for i in range(1, 11):  
    image_path = os.path.join(IMAGES_DIR, f'sudoku{i}.jpg')
    print(f"\nProcessing {image_path}...")

    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"Image {image_path} not found. Skipping.")
            continue

        img = resize_image(img, width=800)
        board, location = locate_board(img)
        gray = cv2.cvtColor(board, cv2.COLOR_BGR2GRAY)
        rois = split_into_boxes(gray)
        rois = np.array(rois).reshape(-1, input_size, input_size, 1)

        prediction = model.predict(rois)
        predicted_numbers = [classes[np.argmax(i)] for i in prediction]
        board_num = np.array(predicted_numbers).astype('uint8').reshape(9, 9)

        print("Detected Sudoku Board:")
        for row in board_num:
            print(row)

        solved_board_nums = get_board(board_num)

        print("\nSolved Sudoku Board:")
        for row in solved_board_nums:
            print(row)

        # Show result
        binArr = np.where(np.array(predicted_numbers) > 0, 0, 1)
        flat_solved_board_nums = solved_board_nums.flatten() * binArr
        mask = np.zeros_like(board)
        solved_board_mask = render_numbers(mask, flat_solved_board_nums)
        inv = inverse_perspective(img, solved_board_mask, location)
        combined = cv2.addWeighted(img, 0.7, inv, 1, 0)

        # Displays the individual steps for each image
        cv2.imshow(f"Original Image {i}", img)
        cv2.imshow(f"Detected Board {i}", board)
        cv2.imshow(f"Solved Board {i}", combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error processing image {image_path}: {e}")

cv2.destroyAllWindows()
