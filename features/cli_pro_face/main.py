import cv2
from deepface import DeepFace

def detect_mood(frame):
    try:
        obj = DeepFace.analyze(frame, actions=['emotion'])
        return max(obj[0]['emotion'], key=obj[0]['emotion'].get)  # returns the emotion with highest score
    except ValueError as e:
        if 'Face could not be detected in numpy array' in str(e):
            print("ValueError: Face could not be detected. Please ensure that the picture is a face photo.")
            return None
        else:
            raise  # re-raise for other unexpected errors

def main():
    cap = cv2.VideoCapture(0)  # capture video from webcamera
    while True:
        _, frame = cap.read()  # read each frame from the stream
        mood = detect_mood(frame)
        if mood is not None:
            print("Detected Mood: ", mood)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to quit
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
