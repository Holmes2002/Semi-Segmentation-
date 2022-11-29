import os 
import cv2 
abc=os.listdir("train")
with open('unlabeled.txt',"w") as f:
    for file in os.listdir("train"):
        png=file.replace("jpg","png")
        if "png" in file or png in abc :
            continue
        img=cv2.imread("train/"+file)
        h,w,_=img.shape
        f.write("train/"+file+"\n")
