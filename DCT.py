import numpy as np
import cv2 as cv
import sys
import pandas as pd
import os

#img = cv.imread(r'C:\Users\shubh\Desktop\sample_images\Malignant'+'\\'+'M_3.jpg')

#imgGray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#imgResize = cv.resize(img, (512,512))
def main():
        start = None
        end = None
        print(start, end)

        if start == None:
            print("Start not specified, setting to 1")
            start = 1

        if end == None:
            print("End not specified, setting to 5")
            end = 1000

        if start > end:
            print("Start > End")
            sys.exit(0)

        opArr4 = []
        opArr8 = []
        opArr16 = []
        apArr_32 = []
        apArr_64 = []

        for x in ("Benign", "Malignant"):
            print(x)
            if x == "Benign":
                fName = "B_"
            if x == "Malignant":
                fName = "M_"

            for f in range(start,end+1):
                print(x + str(f))
                img = cv.imread(r"C:\Users\shubh\Desktop\BE PROJECT\BE seminar\datasets\ISIC_Complete\\" +  x + "\Images\\" + fName + str(f) + ".jpg")

                img = cv.resize(img, (512,512))
                fNameDCT = fName + str(f)
                DctTransform_4, DctTransform_8, DctTransform_16, DctTransform_32, DctTransform_64, reducedImg = dct_transform(img, fNameDCT, x)

                opArr4.append(DctTransform_4)
                opArr8.append(DctTransform_8)
                opArr16.append(DctTransform_16)
                apArr_32.append(DctTransform_32)
                apArr_64.append(DctTransform_64)

        pd.DataFrame(opArr4).to_csv("opDCT_4", mode="a", header=True, index = False)
        pd.DataFrame(opArr8).to_csv("opDCT_8", mode="a", header=True, index = False)
        pd.DataFrame(opArr16).to_csv("opDCT_16", mode="a", header=True, index = False)
        pd.DataFrame(apArr_32).to_csv("opDCT_32", mode='a', header=True, index = False)
        pd.DataFrame(apArr_64).to_csv("opDCT_64", mode='a', header=True, index = False)

def dct_transform(image, fileName, diagnosis):
    red =[]
    green=[]
    blue=[]
    b,g,r = cv.split(image)

    for x in (r,b,g):
        if x is r:
            i = cv.dct(x.astype(np.float64))
            i = cv.convertScaleAbs(i)
            for values in i:
                red.append(values)
        elif x is g:
            i = cv.dct(x.astype(np.float64))
            i = cv.convertScaleAbs(i)
            for values in i:
                green.append(values)
        else:
            i = cv.dct(x.astype(np.float64))
            i = cv.convertScaleAbs(i)
            for values in i:
                blue.append(values)

    red_4=[]
    green_4=[]
    blue_4=[]

    for row in range(4):
        for col in range(4):
            red_4.append(red[row][col])
            green_4.append(green[row][col])
            blue_4.append(blue[row][col])
    DCT_4 = red_4
    DCT_4.extend(green_4)
    DCT_4.extend(blue_4)
    if diagnosis == "Benign":
        DCT_4.append("Benign")
    else:
        DCT_4.append("Malignant")
    print(DCT_4)

    red_8=[]
    green_8=[]
    blue_8=[]
    DCT_8=[]

    for row in range(8):
        for col in range(8):
            red_8.append(red[row][col])
            green_8.append(green[row][col])
            blue_8.append(blue[row][col])
    DCT_8 = red_8
    DCT_8.extend(green_8)
    DCT_8.extend(green_8)
    if diagnosis == "Benign":
        DCT_8.append("Benign")
    else:
        DCT_8.append("Malignant")
    print(DCT_8)

    red_16=[]
    green_16=[]
    blue_16=[]
    DCT_16=[]

    for row in range(16):
        for col in range(16):
            red_16.append(red[row][col])
            green_16.append(green[row][col])
            blue_16.append(blue[row][col])
    DCT_16 = red_16
    DCT_16.extend(green_16)
    DCT_16.extend(blue_16)
    if diagnosis == "Benign":
        DCT_16.append("Benign")
    else:
        DCT_16.append("Malignant")
    print(DCT_16)

    red_32=[]
    green_32=[]
    blue_32=[]
    DCT_32=[]

    for row in range(32):
        for col in range(32):
            red_32.append(red[row][col])
            green_32.append(green[row][col])
            blue_32.append(blue[row][col])
    DCT_32 = red_32
    DCT_32.extend(green_32)
    DCT_32.extend(blue_32)
    if diagnosis == "Benign":
        DCT_32.append("Benign")
    else:
        DCT_32.append("Malignant")
    print(DCT_32)

    red_64=[]
    green_64=[]
    blue_64=[]

    for row in range(64):
        for col in range(64):
            red_64.append(red[row][col])
            green_64.append(green[row][col])
            blue_64.append(blue[row][col])
    DCT_64 = red_64
    DCT_64.extend(green_64)
    DCT_64.extend(blue_64)
    if diagnosis == "Benign":
        DCT_64.append("Benign")
    else:
        DCT_64.append("Malignant")
    print(DCT_64)

    directory = r"C:\Users\shubh\Desktop\BE PROJECT\BE seminar\DCT_reduced\\"+diagnosis
    os.chdir(directory)
    imgGray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    final_image = cv.dct(imgGray.astype(np.float64))
    final_image = cv.convertScaleAbs(final_image)
    fileNameReduce = fileName + "_" +"Reduced.jpg"
    cv.imwrite(fileNameReduce, final_image)


    #cv.imshow('DCT', final_image)

    return DCT_4,DCT_8,DCT_16,DCT_32,DCT_64,final_image

if __name__ == '__main__':
	main()
