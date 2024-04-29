import cv2 as cv

img = cv.imread('/home/user/skdi_dataset/base_dir/val_dir/vasc/ISIC_0025452.jpg')


size = img.shape
st_x, st_y = size[1]//2 - 151, size[0]//2 - 151
end_x, end_y = st_x+300, st_y+300

color = (0, 255, 0)  # Green color in this case

# Define the thickness of the box lines
thickness = 4

n_img = img[st_y:end_y, st_x:end_x]

image_with_box = cv.rectangle(img, (st_x-3, st_y-3), (end_x+3, end_y+3), color, thickness)

cv.imshow('x', img)
cv.imshow('y', n_img)
cv.waitKey()
cv.destroyAllWindows()
