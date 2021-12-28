# MEAN FILTER
import numpy as np
import cv2

imgs = []
for i in range(5):
    imgs.append(cv2.imread("imgs/img" + str(i) + ".jpg"))

kernelDimension = [3, 5, 7, 9]

for i in range(5):
    for y in kernelDimension:
        width, height, channels = imgs[i].shape
        paddingValue = y // 2
        for row in range(paddingValue, width - paddingValue):
            for column in range(paddingValue, height - paddingValue):
                for channel in range(channels):
                    imagePart = imgs[i][row - paddingValue: row + paddingValue + 1,
                                 column - paddingValue: column + paddingValue + 1, channel]
                    imgs[i][row][column][channel] = np.mean(imagePart)
        cv2.imwrite("results/meanFilter/img"+str(i)+"_"+str(y)+".jpg", imgs[i])


# GAUSSAIN FILTER

sigma = [0.5, 1, 1.5, 2]

imgs = []
for i in range(5):
    imgs.append(cv2.imread("imgs/img" + str(i) + ".jpg"))

for i in range(5):
    for kernel in kernelDimension:
        for z in sigma:
            width, height, channels = imgs[i].shape
            paddingValue = kernel // 2
            x, y = np.mgrid[-kernel//2 + 1:kernel//2 + 1, -kernel//2 + 1:kernel//2 + 1]
            g = np.exp(-((x ** 2 + y ** 2) / (2.0 * z ** 2)))
            filter = g / g.sum()
            for row in range(paddingValue, width - paddingValue):
                for column in range(paddingValue, height - paddingValue):
                    for channel in range(channels):
                        imagePart = imgs[i][row - paddingValue: row + paddingValue + 1, column - paddingValue: column + paddingValue + 1, channel]
                        imgs[i][row][column][channel] = np.sum(np.multiply(imagePart, filter))

            cv2.imwrite("results/gaussianFilter/img" + str(i) + "_" + str(kernel) + "_" + str(z) + ".jpg", imgs[i])


#Kuwahara Filter
imgs = []
for i in range(5):
    imgs.append(cv2.imread("imgs/img" + str(i) + ".jpg"))

def rgb2hsv(rgb):
    rgb = rgb.astype('float')
    maxv = np.amax(rgb, axis=2)
    maxc = np.argmax(rgb, axis=2)
    minv = np.amin(rgb, axis=2)
    minc = np.argmin(rgb, axis=2)

    hsv = np.zeros(rgb.shape, dtype='float')
    hsv[maxc == minc, 0] = np.zeros(hsv[maxc == minc, 0].shape)
    hsv[maxc == 0, 0] = (((rgb[..., 1] - rgb[..., 2]) * 60.0 / (maxv - minv + np.spacing(1))) % 360.0)[maxc == 0]
    hsv[maxc == 1, 0] = (((rgb[..., 2] - rgb[..., 0]) * 60.0 / (maxv - minv + np.spacing(1))) + 120.0)[maxc == 1]
    hsv[maxc == 2, 0] = (((rgb[..., 0] - rgb[..., 1]) * 60.0 / (maxv - minv + np.spacing(1))) + 240.0)[maxc == 2]
    hsv[maxv == 0, 1] = np.zeros(hsv[maxv == 0, 1].shape)
    hsv[maxv != 0, 1] = (1 - minv / (maxv + np.spacing(1)))[maxv != 0]
    hsv[..., 2] = maxv

    return hsv

for i in range(5):

    for kernel in kernelDimension:
        rgbImg = imgs[i]
        hsvImg = rgb2hsv(imgs[i])
        width, height, channels = hsvImg.shape
        paddingValue = kernel // 2
        for row in range(paddingValue, width - paddingValue):
            for column in range(paddingValue, height - paddingValue):
                imagePart = hsvImg[row - paddingValue: row + paddingValue + 1, column - paddingValue: column + paddingValue + 1, 2]
                widthPart, heightPart = imagePart.shape
                Q1 = imagePart[0: heightPart // 2 + 1, widthPart // 2: widthPart]
                Q2 = imagePart[0: heightPart // 2 + 1, 0: widthPart // 2 + 1]
                Q3 = imagePart[heightPart // 2: heightPart, 0: widthPart // 2 + 1]
                Q4 = imagePart[heightPart // 2: heightPart, widthPart // 2: widthPart]
                stds = np.array([np.std(Q1), np.std(Q2), np.std(Q3), np.std(Q4)])
                min_std = stds.argmin()

                if min_std == 0:
                    rgbImg[row][column][0] = np.mean(rgbImg[row - heightPart // 2: row + 1, column: column + widthPart // 2 + 1, 0])
                    rgbImg[row][column][1] = np.mean(rgbImg[row - heightPart // 2: row + 1, column: column + widthPart // 2 + 1, 1])
                    rgbImg[row][column][2] = np.mean(rgbImg[row - heightPart // 2: row + 1, column: column + widthPart // 2 + 1, 2])

                elif min_std == 1:
                    rgbImg[row][column][0] = np.mean(rgbImg[row - heightPart // 2:  row + 1, column - widthPart // 2: column + 1, 0])
                    rgbImg[row][column][1] = np.mean(rgbImg[row - heightPart // 2:  row + 1, column - widthPart // 2: column + 1, 1])
                    rgbImg[row][column][2] = np.mean(rgbImg[row - heightPart // 2:  row + 1, column - widthPart // 2: column + 1, 2])

                elif min_std == 2:
                    rgbImg[row][column][0] = np.mean(rgbImg[row: row + heightPart // 2 + 1, column - widthPart // 2:  column + 1, 0])
                    rgbImg[row][column][1] = np.mean(rgbImg[row: row + heightPart // 2 + 1, column - widthPart // 2:  column + 1, 1])
                    rgbImg[row][column][2] = np.mean(rgbImg[row: row + heightPart // 2 + 1, column - widthPart // 2:  column + 1, 2])

                elif min_std == 3:
                    rgbImg[row][column][0] = np.mean(rgbImg[row: row + heightPart // 2 + 1, column: column + heightPart // 2 + 1, 0])
                    rgbImg[row][column][1] = np.mean(rgbImg[row: row + heightPart // 2 + 1, column: column + heightPart // 2 + 1, 1])
                    rgbImg[row][column][2] = np.mean(rgbImg[row: row + heightPart // 2 + 1, column: column + heightPart // 2 + 1, 2])

        cv2.imwrite("results/kuwaharaFilter/img" + str(i) + "_" + str(kernel) + ".jpg", rgbImg)
