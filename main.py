#! python3
import cv2
import numpy as np


def contour(img,area = 50,colour=(0,255,0),thick=2,slice=False):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blank = np.zeros((img.shape[0],img.shape[1],3),np.uint8)
    slices = []

    # adapthreshguass = cv2.Canny(adapthreshguass,0,500)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cont in contours:
        if cv2.contourArea(cont) > area:
            x,y,w,h = cv2.boundingRect(cont)
            if slice:
                slices.append([x,x+w,y,y+h])
            cv2.rectangle(blank,(x,y),(x+w,y+h),colour,thick,)

    if slice:
        return blank, slices
    else:
        return blank


img = cv2.imread("images/img3.jpeg")
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


# gray = cv2.resize(gray,None,fx=0.25,fy=0.25)
# gray = cv2.GaussianBlur(gray, (7, 7), 0)
# gray = cv2.resize(gray,(4000,3000))


# ret, simplethresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# adapthreshmean = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV, 99, -5)
adapthreshguass = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY, 99, -5)


adapthreshguass = cv2.erode(adapthreshguass,(5,5),iterations=1)


# adapthreshguass = cv2.bilateralFilter(adapthreshguass, 9, 75, 75)
adapthreshguass = cv2.medianBlur(adapthreshguass, 7)
# adapthreshguass = cv2.GaussianBlur(adapthreshguass, (7, 7), 0)

blank = contour(adapthreshguass,area= 250,colour=(255,255,255),thick=-1)
blank, slices = contour(blank,area=250,slice=True)


# simplethresh = cv2.resize(simplethresh,None,fx=0.25,fy=0.25)
# adapthreshmean = cv2.resize(adapthreshmean,None,fx=0.25,fy=0.25)
adapthreshguass = cv2.resize(adapthreshguass,None,fx=0.25,fy=0.25)
blank = cv2.resize(blank,None,fx=0.25,fy=0.25)


# cv2.imshow("simplethresh",simplethresh)
# cv2.imshow("adapthreshmean",adapthreshmean)
# cv2.imshow("adapthreshguass",adapthreshguass)
# cv2.imshow("blank",blank)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


for s in slices:
    IMG = img[s[2]:s[3],s[0]:s[1]]
    # IMG = cv2.resize(IMG,None,fx=0.25,fy=0.25)
    cv2.imshow("IMG",IMG)
    if cv2.waitKey(2000) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()
