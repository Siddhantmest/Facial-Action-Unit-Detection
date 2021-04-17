import numpy as np
import dlib
import matplotlib.pyplot as plt
import imutils
import cv2
import os

PREDICTOR_PATH = "./data/shape_predictor_68_face_landmarks.dat"

txt_file = open('./test2/examples.txt','r')
lines = txt_file.readlines()
new_paths = []
for ps in lines:
    new_paths.append(ps.replace('\r', '').replace('\n', ''))
ref_path = new_paths[1]
new_paths.remove(ref_path)
file_path = './test2_1'


def procrustes_analysys(A, B):
    """Procrustes analysis
    Basic algorithm is
        1. Recenter the points based on their mean: compute a mean and subtract it from every points in shape
        2. Normalize
        3. Rotate one of the shapes and find MSE
    Args:
        A:
        B:
    Returns:
    """
    h_A, w_A = A.shape
    h_B, w_B = B.shape

    # compute mean of each A and B
    Amu = np.mean(A, axis=0)
    Bmu = np.mean(B, axis=0)

    # subtract a mean
    A_base = A - Amu
    B_base = B - Bmu

    # normalize
    ssum_A = (A_base**2).sum()
    ssum_B = (B_base**2).sum()

    norm_A = np.sqrt(ssum_A)
    norm_B = np.sqrt(ssum_B)

    normalized_A = A_base / norm_A
    normalized_B = B_base / norm_B

    if (w_B < w_A):
        normalized_B = np.concatenate((normalized_B, np.zeros(h_A, w_A - w_B)), 0)

    A = np.dot(normalized_A.T, normalized_B)

    # SVD
    u, s, vh = np.linalg.svd(A, full_matrices=False)
    v = vh.T
    T = np.dot(v, u.T)
    
    scale = norm_A / norm_B
   
    return T, scale

def shape_to_np(shape, dtype="int"):
    """Take a shape object and convert it to numpy array
    Args:
        shape: an object returned by dlib face landmark detector containing the 68 (x, y)-coordinates of the facial landmark regions
        dtype: int
    Returns:
        coords: (68,2) numpy array
    """
    coords = np.zeros((68, 2), dtype=dtype)

    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    return coords

def get_face(img1_detection):
    for face in img1_detection:
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
    
        # draw box over face
        cv2.rectangle(img1, (x,y), (x+w,y+h), (0,255,0), 2)
        
    img_height, img_width = img1.shape[:2]
    cv2.putText(img1, "HOG", (img_width-50,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0,255,0), 2)
    
    # display output image
    plt.imshow(img1)
    
def plot_landmarks():
    # plot facial landmarks on the image
    for (x, y) in img1_shape:
        cv2.circle(img1, (x, y), 1, (0, 0, 255), -1)
        
    plt.imshow(img1)
    

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)

for path in new_paths:
    img1 = dlib.load_rgb_image(ref_path)
    img2 = dlib.load_rgb_image(path)
    
    img1_detection = detector(img1, 1)
    img2_detection = detector(img2, 1)
    
    img1_shape = predictor(img1, img1_detection[0])
    img2_shape = predictor(img2, img2_detection[0])
    
    img1_shape1 = shape_to_np(img1_shape)
    img2_shape1 = shape_to_np(img2_shape)
    
    M, scale = procrustes_analysys(img1_shape1, img2_shape1)
    theta = np.rad2deg(np.arccos(M[0][0]))
    #print("theta is {}".format(theta))
    
    rotation_matrix = cv2.getRotationMatrix2D((img1.shape[1]/2, img1.shape[0]/2), theta, 1)
    dst = cv2.warpAffine(img2, rotation_matrix, (img2.shape[1], img2.shape[0]))
    
    
    img2_aligned = dlib.get_face_chip(dst, img2_shape)
    
    
    img2_aligned_resized = imutils.resize(img2_aligned, width = img2.shape[0], height = img2.shape[1])
    
    
    img_name = path.split('/')[-1].split(".")[0] + '.jpg'
    cv2.imwrite(os.path.join(file_path , img_name), img2_aligned_resized)
    
    if path == new_paths[-1]:
        img1_aligned = dlib.get_face_chip(img1, img1_shape)
        img1_aligned_resized = imutils.resize(img1_aligned, width = img1.shape[0], height = img1.shape[1])
        img_name = ref_path.split('/')[-1].split(".")[0] + '.jpg'
        cv2.imwrite(os.path.join(file_path , img_name), img1_aligned_resized)