import cv2 as cv
import lab1_filters as lf

img = cv.imread("12.jpeg")
cv.imshow("Image", img)
cv.imshow("Gray Image", lf.toShapesOfGray(img))
cv.imshow("200x200 Image", lf.changeSize(img, 200, 200))
cv.imshow("Sepia Image", lf.toSepia(img))
cv.imshow("Vignette Image", lf.vignette(img, 450))
cv.imshow("Pixelizied Image", lf.pixelization(img, 12, 0, 0, 400, 400))
cv.waitKey(0)
cv.destroyAllWindows()