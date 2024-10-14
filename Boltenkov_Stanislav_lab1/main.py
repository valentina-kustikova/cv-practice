import filterscv

def main():
    imagePath = input()
    numOperations = int(input())
    if (numOperations == 0):
        filterscv.conversionToGray(imagePath)
    elif (numOperations == 1):
        kx, ky = [float(i) for i in (input().split())]
        filterscv.resizeImage(imagePath, kx, ky)
    elif (numOperations == 2):
        filterscv.sepiaPhotoEffect(imagePath)
    elif (numOperations == 3):
        filterscv.vignettePhotoEffect(imagePath)
    elif (numOperations == 4):
        x0, y0, x1, y1 = [int(i) for i in (input().split())]
        filterscv.pixelation(imagePath, x0, y0, x1, y1)
        
main()