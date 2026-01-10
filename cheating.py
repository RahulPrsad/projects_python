import cv2
import mediapipe as mp
import math

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

violation_count = 0

def eye_aspect_ratio(landmarks, left_indices, right_indices, frame_width, frame_height):
    # Get coordinates
    def coords(i):
        lm = landmarks[i]
        return int(lm.x * frame_width), int(lm.y * frame_height)
    
    # Left eye
    lx1, ly1 = coords(left_indices[0])
    lx2, ly2 = coords(left_indices[1])
    lx3, ly3 = coords(left_indices[2])
    lx4, ly4 = coords(left_indices[3])
    
    # Vertical distance
    left_ear = math.hypot(lx2 - lx4, ly2 - ly4)
    
    # Right eye
    rx1, ry1 = coords(right_indices[0])
    rx2, ry2 = coords(right_indices[1])
    rx3, ry3 = coords(right_indices[2])
    rx4, ry4 = coords(right_indices[3])
    
    right_ear = math.hypot(rx2 - rx4, ry2 - ry4)
    
    return (left_ear + right_ear) / 2

# Eye landmark indices (from MediaPipe documentation)
LEFT_EYE = [33, 159, 145, 133]
RIGHT_EYE = [362, 386, 374, 263]

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(frame_rgb)

    frame_height, frame_width = frame.shape[:2]

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        # Eye aspect ratio (simple check if eyes are closed or looking away)
        ear = eye_aspect_ratio(landmarks, LEFT_EYE, RIGHT_EYE, frame_width, frame_height)
        
        # Head movement - check nose landmark x,y
        nose_x = int(landmarks[1].x * frame_width)
        nose_y = int(landmarks[1].y * frame_height)

        if ear < 5:  # Threshold for eye looking away / closed
            cv2.putText(frame, "EYES AWAY!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            violation_count += 1
        elif nose_x < frame_width*0.2 or nose_x > frame_width*0.8:  # Head turned left/right
            cv2.putText(frame, "HEAD TURNED!", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            violation_count += 1
        else:
            cv2.putText(frame, "LOOKING OK", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "NO FACE DETECTED!", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        violation_count += 1

    # Display violation count
    cv2.putText(frame, f"Violations: {violation_count}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Exam Proctoring", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
