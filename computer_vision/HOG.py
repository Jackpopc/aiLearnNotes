import cv2
import numpy as np
import math
import matplotlib.pyplot as plt

img = cv2.imread("../data/2007_000129.jpg", cv2.IMREAD_GRAYSCALE)


def compute_image_gradient(img):
    x_values = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
    y_values = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)
    magnitude = abs(cv2.addWeighted(x_values, 0.5, y_values, 0.5, 0))
    angle = cv2.phase(x_values, y_values, angleInDegrees=True)
    return magnitude, angle


def choose_bins(gradient_angle, unit, bin_size):
    idx = int(gradient_angle / unit)
    mod = gradient_angle % unit
    return idx, (idx + 1) % bin_size, mod


def compute_cell_gradient(cell_magnitude, cell_angle, bin_size, unit):
    centers = [0] * bin_size
    for i in range(cell_magnitude.shape[0]):
        for j in range(cell_magnitude.shape[1]):
            strength = cell_magnitude[i][j]
            gradient_angle = cell_angle[i][j]
            min_angle, max_angle, mod = choose_bins(gradient_angle, unit, bin_size)
            print(gradient_angle, unit, min_angle, max_angle)
            centers[min_angle] += (strength * (1 - (mod / unit)))
            centers[max_angle] += (strength * (mod / unit))
    return centers


def normalized(cell_gradient_vector):
    hog_vector = []
    for i in range(cell_gradient_vector.shape[0] - 1):
        for j in range(cell_gradient_vector.shape[1] - 1):
            block_vector = []
            block_vector.extend(cell_gradient_vector[i][j])
            block_vector.extend(cell_gradient_vector[i][j + 1])
            block_vector.extend(cell_gradient_vector[i + 1][j])
            block_vector.extend(cell_gradient_vector[i + 1][j + 1])
            mag = lambda vector: math.sqrt(sum(i ** 2 for i in vector))
            magnitude = mag(block_vector)
            if magnitude != 0:
                normalize = lambda block_vector, magnitude: [element / magnitude for element in block_vector]
                block_vector = normalize(block_vector, magnitude)
            hog_vector.append(block_vector)
    return hog_vector


def visual(cell_gradient, height, width, cell_size, unit):
    feature_image = np.zeros([height, width])
    cell_width = cell_size / 2
    max_mag = np.array(cell_gradient).max()
    for x in range(cell_gradient.shape[0]):
        for y in range(cell_gradient.shape[1]):
            cell_grad = cell_gradient[x][y]
            cell_grad /= max_mag
            angle = 0
            angle_gap = unit
            for magnitude in cell_grad:
                angle_radian = math.radians(angle)
                x1 = int(x * cell_size + magnitude * cell_width * math.cos(angle_radian))
                y1 = int(y * cell_size + magnitude * cell_width * math.sin(angle_radian))
                x2 = int(x * cell_size - magnitude * cell_width * math.cos(angle_radian))
                y2 = int(y * cell_size - magnitude * cell_width * math.sin(angle_radian))
                cv2.line(feature_image, (y1, x1), (y2, x2), int(255 * math.sqrt(magnitude)))
                angle += angle_gap
    return feature_image


def main(img):
    cell_size = 16
    bin_size = 9
    unit = 360 // bin_size
    height, width = img.shape

    magnitude, angle = compute_image_gradient(img)

    cell_gradient_vector = np.zeros((height // cell_size, width // cell_size, bin_size))
    for i in range(cell_gradient_vector.shape[0]):
        for j in range(cell_gradient_vector.shape[1]):
            cell_magnitude = magnitude[i * cell_size:(i + 1) * cell_size,
                             j * cell_size:(j + 1) * cell_size]
            cell_angle = angle[i * cell_size:(i + 1) * cell_size,
                         j * cell_size:(j + 1) * cell_size]
            cell_gradient_vector[i][j] = compute_cell_gradient(cell_magnitude, cell_angle, bin_size, unit)
    hog_vector = normalized(cell_gradient_vector)
    hog_image = visual(cell_gradient_vector, height, width, cell_size, unit)
    plt.imshow(hog_image, cmap=plt.cm.gray)
    plt.show()


if __name__ == '__main__':
    img = cv2.imread('../data/2007_002293.jpg', cv2.IMREAD_GRAYSCALE)
    cv2.imshow("origin", img)
    cv2.waitKey()
    main(img)
