import cv2
import numpy as np

# Load the image
image = cv2.imread('hair.jpeg')
if image is not None:
  # Resizing
  down_width = 500
  down_height = 500
  down_points = (down_width, down_height)
  img = cv2.resize(image, down_points, interpolation= cv2.INTER_LINEAR)

  # Display the result
  cv2.imshow("Original Image",img)

  roi = img[100:400,100:400]
#   print(roi)

  cv2.imshow("Region of Interest",roi)
  cv2.waitKey(0)

  # Convert the original image to grayscale
  grayScale = cv2.cvtColor( roi, cv2.COLOR_RGB2GRAY )

  # Kernel for the morphological filtering
  kernel = cv2.getStructuringElement(1,(17,17))

  # Perform the blackHat filtering on the grayscale image to find the hair countours
  blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

  # intensify the hair countours in preparation for the inpainting algorithm
  ret,thresh2 = cv2.threshold(blackhat,10,255,cv2.THRESH_BINARY)
  print( thresh2.shape )

  # inpaint the original image depending on the mask
  dst = cv2.inpaint(roi,thresh2,1,cv2.INPAINT_TELEA)
  cv2.imshow("Hair Removed",dst)
  cv2.imwrite('/home/vipul/Vipul/ML/Skin Cancer Diagnosis/Skin-Cancer-Classification-using-CNN-/skin_image.jpg', dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

else:
    print("Error loading the image.")