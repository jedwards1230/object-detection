import cv2
from imutils.video import FPS

video_stream = "http://192.168.1.143:56000/mjpeg"

if __name__=='__main__':
    vcap = cv2.VideoCapture(video_stream)
    
    while True:
        success, img = vcap.read()
        if img is not None:
            img = cv2.flip(img, 1)
            fps = FPS().start()
            H, W, C = img.shape
            
            fps.update()
            fps.stop()
            fps_text = f'FPS: {round(fps.fps(), 2)}'
            cv2.putText(img, fps_text, (10, H - (20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            cv2.imshow('Image', img)
            
            key = cv2.waitKey(1)
            if key == ord("q"):
                break