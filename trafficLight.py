import cv2
import imutils
import yaml
import os, os.path

#yaml fájl betöltése
with open('../DTLD_Labels/Bochum_all.yml') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)
    #print(yaml.dump(data[1]))

#######

#képek lista betöltése
imgs = []
path = "../scratch/fs2/DTLD_final/Bochum/Bochum1/2015-04-21_17-09-21"
valid_images = [".tiff"]
for f in os.listdir(path):
    ext = os.path.splitext(f)[1]
    if ext.lower() not in valid_images:
        continue

    # imgs.append(Image.open(os.path.join(path,f)))
    imgs.append(cv2.imread(os.path.join(path, f)))
(h, w, d) = imgs[0].shape
print("width={}, height={}, depth={}".format(w, h, d))

cv2.imshow('image',imgs[0])

(B, G, R) = imgs[0][100, 50]
print("R={}, G={}, B={}".format(R, G, B))
output = imgs[0].copy()
for i in data[0].get("objects"):

    tl = imgs[0][i.get("y"):i.get("y") + i.get("height"),
         i.get("x"):i.get("x") + i.get("width")]
    #cv2.imshow(i.get("track_id"), tl)


    cv2.rectangle(output, (i.get("x"), i.get("y")), (i.get("x") + i.get("width"), i.get("y") + i.get("height")), (0, 0, 255), 2)
cv2.imshow("Rectangle", output)
cv2.waitKey(0)

n =2
print(n)