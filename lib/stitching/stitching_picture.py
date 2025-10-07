import cv2
image_basename = "img"
images_location = [f"images/{image_basename}{x}.jpg" for x in range(7)]
images = [cv2.imread(img) for img in images_location]

stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
status, panorama = stitcher.stitch(images)

if status == cv2.Stitcher_OK:
    print("Stitching was a success!")
    cv2.imwrite("drone_panorama.jpg", panorama)
else:
    print("Stitching failed with error code:", status)
    
