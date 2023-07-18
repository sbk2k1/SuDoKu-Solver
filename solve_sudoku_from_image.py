import cv2
import numpy as np
from PIL import Image
from keras.models import load_model
model = load_model('Digit Classification.h5')

def preprocess(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
    blur_gray_image = cv2.GaussianBlur(gray_image, (3,3),6) 
    final_image = cv2.adaptiveThreshold(blur_gray_image,255,1,1,11,2)
    return final_image

def main_outline(contour):
    biggest = np.array([])
    max_area = 0
    for i in contour:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i , 0.02* peri, True)
            if area > max_area and len(approx) ==4:
                biggest = approx
                max_area = area
    return biggest ,max_area

def reframe(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4,1,2),dtype = np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]
    points_new[3] = points[np.argmax(add)]
    diff = np.diff(points, axis =1)
    points_new[1] = points[np.argmin(diff)]
    points_new[2] = points[np.argmax(diff)]
    return points_new

def splitcells(img):
    # print("image is ->", img)
    rows = np.vsplit(img,9)
    boxes = []
    for r in rows:
        cols = np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def CropCell(cells):
    Cells_croped = []
    for image in cells:
        
        img = np.array(image)
        img = img[4:46, 6:46]
        img = Image.fromarray(img)
        Cells_croped.append(img)
        
    return Cells_croped

def read_cells(cell,model):
    result = []
    for image in cell:
        # preprocess the image as it was in the model 
        img = np.asarray(image)
        img = img[4:img.shape[0] - 4, 4:img.shape[1] -4]
        img = cv2.resize(img, (32, 32))
        img = img / 255
        img = img.reshape(1, 32, 32, 1)
        # getting predictions and setting the values if probabilities are above 65% 
        
        pred = model.predict(img)[0]
        max_pred = pred[np.argmax(pred)]
        if max_pred > 0.65:
            result.append(np.argmax(pred))
        else:
            result.append(0)
    return result

def get_grid(puzzle):
    # read puzzle as cv2 image from file
    puzzle = cv2.imread(puzzle)
    puzzle = cv2.resize(puzzle, (450,450))
    # Preprocessing Puzzle 
    su_puzzle = preprocess(puzzle)
    su_imagewrap = ""

    # Finding the outline of the sudoku puzzle in the image
    #su_contour_1= su_puzzle.copy()
    su_contour, hierarchy = cv2.findContours(su_puzzle,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

    #black_img = np.zeros((450,450,3), np.uint8)
    su_biggest, su_maxArea = main_outline(su_contour)
    if su_biggest.size != 0:
        su_biggest = reframe(su_biggest)
        #cv2.drawContours(su_contour_2,su_biggest,-1, (0,255,0),10)
        su_pts1 = np.float32(su_biggest)
        su_pts2 = np.float32([[0,0],[450,0],[0,450],[450,450]])
        su_matrix = cv2.getPerspectiveTransform(su_pts1,su_pts2)  
        su_imagewrap = cv2.warpPerspective(puzzle,su_matrix,(450,450))
        su_imagewrap =cv2.cvtColor(su_imagewrap, cv2.COLOR_BGR2GRAY)
    sudoku_cell = splitcells(su_imagewrap)
    sudoku_cell_croped= CropCell(sudoku_cell)
    grid = read_cells(sudoku_cell_croped, model)
    grid = np.asarray(grid)
    grid = np.reshape(grid,(9,9))
    return grid

def next_box(quiz):
    for row in range(9):
        for col in range(9):
            if quiz[row][col] == 0:
                return (row, col)
    return False

#Function to fill in the possible values by evaluating rows collumns and smaller cells

def possible (quiz,row, col, n):
    #global quiz
    for i in range (0,9):
        if quiz[row][i] == n and row != i:
            return False
    for i in range (0,9):
        if quiz[i][col] == n and col != i:
            return False
        
    row0 = (row)//3
    col0 = (col)//3
    for i in range(row0*3, row0*3 + 3):
        for j in range(col0*3, col0*3 + 3):
            if quiz[i][j]==n and (i,j) != (row, col):
                return False
    return True

#Recursion function to loop over untill a valid answer is found. 

def solve(quiz):
    val = next_box(quiz)
    if val is False:
        return True
    else:
        row, col = val
        for n in range(1,10): #n is the possible solution
            if possible(quiz,row, col, n):
                quiz[row][col]=n
                if solve(quiz):
                    return True 
                else:
                    quiz[row][col]=0
        return 
    
def Solved(quiz):
    grid=[]
    for row in range(9):
        if row % 3 == 0 and row != 0:
            #print("....................")
            pass

        for col in range(9):
            if col % 3 == 0 and col != 0:
                #print("|", end=" ")
                pass

            if col == 8:
                grid.append(quiz[row][col])
            else:
                grid.append(str(quiz[row][col]))
    
    #convert grid to a 9x9 matrix
    grid = np.reshape(grid,(9,9))
    return grid
                
            
def solve_sudoku(image):
    grid = get_grid(image)
    if solve(grid):
        return Solved(grid)
    else:
        return "No Solution found. Model may have misread the puzzle. Please try to take a clearer picture."
