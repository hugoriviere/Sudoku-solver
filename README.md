# **UNMAINTAINED**

This program was written for a school assignment and I have no plans to maintain it going forward.

# Sudoku Solver with OCR and Image Processing

## 1. Project Overview and Results
This project implements a Sudoku solver that detects, extracts, and solves Sudoku puzzles from images. By leveraging image processing, convolutional neural networks (CNNs) for Optical Character Recognition (OCR), and backtracking algorithms, the program achieves end-to-end functionality from image input to displaying the solved puzzle.  

### Key Objectives
- Automate the detection and solving of Sudoku puzzles from real-world images.
- Ensure high accuracy in digit recognition using a pre-trained CNN model.
- Provide a visually intuitive overlay of the solution on the original image.

### Key Results
- Achieved an OCR digit recognition accuracy of **96%**.
- Successfully processed images of varying quality and sizes, resizing them dynamically for optimal performance.

---

## 2. Source Code
The source code is organized as follows:

```plaintext
sudoku-solver/
│
├── solver.py                      # Backtracking algorithm for solving Sudoku puzzles
├── main.py                        # Main script to process images and run the pipeline
├── image/
│   ├── sudoku1.jpg     
│   ├── sudoku2.jpg
│   ├── sudoku3.jpg
|   ├── sudoku4.jpg
│   ├── sudoku5.jpg     
│   ├── sudoku6.jpg
│   ├── sudoku7.jpg
|   ├── sudoku8.jpg
│
├── model/
│   ├──  model-OCR.h5              # Pre-trained OCR model for digit recognition
|
├── requirements.txt               # Required Python packages
└── README.md                      # Project documentation
```



### Dependencies
- Python 3.8+
- OpenCV
- TensorFlow/Keras
- NumPy
- Imutils

---

## 3. Performance Metrics
| **Metric**                  | **Value**           |
|-----------------------------|---------------------|
| OCR Accuracy                | 96%                 |
| Average Processing Time     | 1.85 s              |
| Memory Usage (Peak)         | 8.5 MB              |
| Supported Image Sizes       | Up to 1920x1080     |

### Performance Analysis
- The model shows robust accuracy on clean images but struggles slightly with blurred or noisy inputs.
- The resizing and preprocessing pipeline significantly improves the OCR performance.

---

## 4. Installation and Usage

### Installation

Clone the repository from GitHub

Installing dependencies:
pip install -r requirements.txt

if the program doesn't work, there's probably a problem with the access paths. I advise you to set your absolute path for images and the ORC model.

to run the code you need to run main in your console.

## 5. References and Documentation

### Libraries Used
- [OpenCV](https://opencv.org/): Used for image processing tasks, including contour detection, resizing, and perspective transformation.
- [TensorFlow/Keras](https://www.tensorflow.org/): Used to build, train, and load the CNN model for Optical Character Recognition (OCR).
- [Imutils](https://pypi.org/project/imutils/): A library that provides convenience functions for image processing, especially for contour operations.

 ### other sources that contributed to my project
- (https://openclassrooms.com/forum/sujet/reconnaissance-de-chiffres-sur-une-image-opencv) : this document explains how to detect digits in an image.
- (https://nsi.xyz/projets/solveur-de-sudoku-en-python/) : this document explains how to solve a sudoku with python

### Algorithm Documentation
- **Backtracking Sudoku Solver**: This is a recursive algorithm that checks possible values in empty cells and backtracks when a cell is unsolvable. It ensures all constraints of Sudoku (each row, column, and 3x3 box containing unique numbers) are respected.
- **OCR Pipeline**: Using a pre-trained Convolutional Neural Network (CNN) model, the pipeline recognizes digits within individual cells of the Sudoku puzzle. Each cell is processed through image transformations before digit prediction to enhance recognition accuracy.

---

## 6. Known Issues and Contributions

### Known Issues
- **Low Accuracy on Noisy Images**: The OCR model struggles to accurately recognize digits on images with high noise, distortion, or poor lighting conditions.
- **Slow Processing for High-Resolution Images**: For images exceeding 1920x1080 resolution, the preprocessing and OCR steps slow down considerably.
- **Limited Support for Non-Standard Puzzles**: Currently, only standard 9x9 Sudoku grids are supported.

### Contributing
We welcome contributions to improve and extend the functionality of this project. Here are some ways you can contribute:

1. **Report Bugs**: If you encounter issues, please create an issue on the GitHub repository with a detailed description.
2. **Submit Enhancements**: If you have code improvements, bug fixes, or new features, submit a pull request. Make sure to include details about your changes.
3. **Feature Requests**: Feel free to open issues to suggest new features or enhancements.

---

## 7. Future Work

### Potential Improvements
- **Improve OCR Accuracy**: Extend the training dataset with a wider variety of fonts and noisy images to improve the OCR model’s robustness in various conditions.
- **Add Real-Time Processing**: Implement a real-time Sudoku detection and solving feature that utilizes webcam input, allowing users to capture puzzles directly.
- **Develop an Interactive GUI**: Create a graphical user interface that allows users to upload images, view intermediate processing stages, and see the solved Sudoku overlaid on the original image.
- **Extend Support for Larger Grids**: Modify the solver to handle larger Sudoku grids, such as 16x16 or 25x25, expanding the range of puzzles that can be solved.

---

