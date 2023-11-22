import numpy as np
import cv2
image=cv2.imread('/Users/guest-user/PycharmProjects/pythonProject2/imagepresent3.jpg')
height, width = image.shape[:2]

# Create a mask of ones and multiply by 255 to make it white
mask = np.ones((height, width), dtype=np.uint8) * 255

# Convert the mask to a grayscale image
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

def click_event(event,x,y,flags,param):
    if event==cv2.EVENT_LBUTTONDOWN:
        print(x,",",y)
        font=cv2.FONT_HERSHEY_SIMPLEX
        strxy=str(x)+","+str(y)
        cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 0), -1)
        cv2.rectangle(mask, (x - 2, y - 2), (x + 2, y + 2), (0, 0, 0), -1)
        cv2.imshow('image',img)

img=cv2.imread('/Users/guest-user/PycharmProjects/pythonProject2/imagepresent3.jpg')

cv2.imshow('image',img)

cv2.setMouseCallback('image',click_event)

cv2.waitKey(30000)
cv2.destroyAllWindows()

cv2.imshow('mask',mask)

cv2.waitKey(10000)
cv2.imwrite('/Users/guest-user/Downloads/mask12frompycharm.jpg', mask)
cv2.destroyAllWindows()