import cv2

CAM_ID = 0
view = cv2.VideoCapture(0)
view.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
view.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
while True:
    ret, frame = view.read()
    cv2.imshow("VideoFrame", frame)
    if cv2.waitKey(1) > 0:
        cv2.imwrite('Sign Language data/sample17.png',
                    frame, params=[cv2.IMWRITE_PNG_COMPRESSION,0])
        break

view.release()
cv2.destroyAllWindows()
