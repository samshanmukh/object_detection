import cv2, sys

picture = sys.argv[1]
casc = sys.argv[2]

# Create a Cascade Classifier
obj_cascade = cv2.CascadeClassifier(casc)

# Reading the image as it is and convert in numpy array
img = cv2.imread(picture)

# Reading the image as gray scale image
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Search the co-ordinates of the image
objects = obj_cascade.detectMultiScale(
	gray_img, 
	scaleFactor = 1.05, 
	minNeighbors = 20, 
	minSize = (30, 30)
)

for x,y,w,h in objects:
	img = cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 3)
	# font = cv2.FONT_HERSHEY_SIMPLEX
	# img = cv2.putText(img, 'Human', (x+w, y+h), font, 0.5, (0,255,0), 3, cv2.LINE_AA)

# Resize the image in width, height 
# resize_img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))

cv2.imshow('detected objects: {}'.format(len(objects)), img)
print(type(objects))
print(objects)

# Press any key to exit or set time as 2000 i.e., miliseconds
cv2.waitKey(0)

cv2.destroyAllWindows()