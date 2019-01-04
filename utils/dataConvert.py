

import os
import cv2


def main():
        basePath = '/media/rodrigo/Rodrigo/PAO/VGG-BdBxs/data/valid/'
        path1 = basePath + 'Pedestrian/'
        path2 = basePath + 'NotPedestrian/'
        path3 = '/media/rodrigo/Rodrigo/PAO/PedestrianTracking-V3/data/valid/pedestrian/'
        path4 = '/media/rodrigo/Rodrigo/PAO/PedestrianTracking-V3/data/valid/notpedestrian/'
        i=1
        for root, dirs, files in os.walk(path1):
            for f in files:
                n = f.split('.')[0]
                img = path1 + f
                #print(img)
                i+=1
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(path3 + str(n) + '.jpg', resized_image)
                print(i)
        i=1
        for root, dirs, files in os.walk(path2):
            for f in files:
                n = f.split('.')[0]
                img = path2 + f
                i+=1
                image = cv2.imread(img)
                resized_image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite(path4 + str(n) + '.jpg', resized_image)
                print(i)

if __name__ == '__main__':
	main() 
