import numpy as np
from PIL import Image

def convertToImage(filename, img_mode):
    file = open(filename)
    lines = file.readlines()
    matrix = []
    for line in lines:
        processed_line = line.replace("\n", "")
        splitted_line = processed_line.split(',')
        row = []
        for number in splitted_line:
            row.append(float(number))
        matrix.append(row)
    npmatrix = np.array([matrix[i] for i in range(len(matrix))], dtype=np.uint8)
    print(npmatrix)
    img = Image.fromarray(npmatrix, img_mode)
    return img

def computeActivation(filename):
    file = open(filename)
    lines = file.readlines()
    matrix = []
    for line in lines:
        processed_line = line.replace("\n", "")
        splitted_line = processed_line.split(',')
        row = []
        for number in splitted_line:
            row.append(float(number))
        matrix.append(row)
        
    indices = []
    for row_number, row in enumerate(matrix):
        for column_number, element in enumerate(row):
            if element == 1:
                indices.append({'x': column_number, 'y': row_number})
        
    return indices

filename1 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/matrix1.txt"
img_mode1 = "1"
img1 = convertToImage(filename1, img_mode1)
img1.show()

filename2 = "/Users/shawnfan/active_learning_prototype/prototype/src/assets/matrix2.txt"
img_mode2 = "RGB"
img2 = convertToImage(filename2, img_mode2)
#img2.show()

print(computeActivation(filename1))

