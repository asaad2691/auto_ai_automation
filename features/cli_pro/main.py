import cv2
from deepface import DeepFace
import os

def detect_emotion(image):
    # Join current directory with image filename
    image_path = os.path.join(os.getcwd(), image)
    try:
        analyzed = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        # Handle DeepFace returning a list or dict
        if isinstance(analyzed, list):
            analyzed = analyzed[0]
        print("Detected emotion : ", analyzed['dominant_emotion'])
        
    except Exception as e:
        print('Error: ', str(e))

def detect_emotion_live():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        return
    print("Press 'q' to quit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            # Handle DeepFace returning a list or dict
            if isinstance(result, list):
                result = result[0]
            emotion = result['dominant_emotion']
            cv2.putText(frame, f'Emotion: {emotion}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        except Exception as e:
            cv2.putText(frame, 'No face detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        cv2.imshow('Live Emotion Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

def main():
    # detect_emotion('features\cli_pro\image.jpg')
    detect_emotion_live()

if __name__ == '__main__':
    # Uncomment the following line to use live webcam emotion detection
    main()
