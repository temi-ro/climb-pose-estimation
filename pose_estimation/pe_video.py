from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp

import numpy as np
import cv2

def get_limbs(pose_landmarks):
    return {
        "left_hand": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST],
        "right_hand": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST],
        "left_elbow": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW],
        "right_elbow": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW],
        "left_foot": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ANKLE],
        "right_foot": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ANKLE],
        "left_shoulder": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER],
        "right_shoulder": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER],
        "left_knee": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_KNEE],
        "right_knee": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_KNEE],
        "left_hip": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_HIP],
        "right_hip": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_HIP],
        "head": pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.NOSE]
    }

# Draw the landmarks on the frame with joints connected
def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
        landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks_list.landmark 
    ])
    solutions.drawing_utils.draw_landmarks(
        annotated_image,
        pose_landmarks_proto,
        solutions.pose.POSE_CONNECTIONS,
        solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

# Draw the connections between the landmarks only if the landmarks are visible
def draw_connections(frame, results, connections):
    landmarks = results.pose_landmarks.landmark

    for connection in connections:
        start_landmark = landmarks[connection[0]]
        end_landmark = landmarks[connection[1]]
        if start_landmark.visibility > 0.5 and end_landmark.visibility > 0.5:
            start_x = int(start_landmark.x * frame.shape[1])
            start_y = int(start_landmark.y * frame.shape[0])
            end_x = int(end_landmark.x * frame.shape[1])
            end_y = int(end_landmark.y * frame.shape[0])
            cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 0), 2)

    return frame

def create_detector():
    detector = solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    return detector

def run_pose_estimation(detector, video_path):
    # Open video capture from webcam (you can replace 0 with the video file path)
    cap = cv2.VideoCapture(video_path)
    prev_landmark = None
    prev_time = 0
    pause = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Get time between frames
        cur_time = cv2.getTickCount()
        time_diff = (cur_time-prev_time)/cv2.getTickFrequency()
        prev_time = cur_time

        # Convert the frame to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if pause:
            key = cv2.waitKey(0)
            if key == ord('q'):
                break
            elif key == 32: # Space bar
                pause = False

        # Process the frame for pose estimation
        results = detector.process(frame)
        if results.pose_landmarks:
            # Draw landmarks on the frame
            # frame_with_landmarks = draw_landmarks_on_image(frame, results)
            draw_connections(frame, results, solutions.pose.POSE_CONNECTIONS)
            
            # Draw the center of gravity
            draw_center_of_gravity(frame, results.pose_landmarks)
            
            # Draw the tension of each limb
            draw_tension(frame, results.pose_landmarks)

            # Compute the speed of the feet
            # if prev_landmark:
            #     print(getFeetSpeed(results.pose_landmarks, prev_landmark, time_diff))
            # prev_landmark = results.pose_landmarks
            
            # Display the frame with landmarks
            cv2.imshow('Pose Estimation', cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        # Pause the video


        # Break the loop if 'q' is pressed
        key = cv2.waitKey(1)
        if key == ord('q'):
            break
        elif key == 32: # Space bar
            pause = not pause
        
    # Release the VideoCapture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

def distance_limbs(limb1, limb2):
    return np.sqrt((limb1.x - limb2.x)**2 + (limb1.y - limb2.y)**2)

# Compute the speed of the feet
# Status: Not working
# Units are in pixels per second, how to get it in meters per second?
def get_feet_speed(cur_landmark, prev_landmark, time_diff):
    limbs = get_limbs(cur_landmark)
    prev_limbs = get_limbs(prev_landmark)

    left_foot = limbs["left_foot"]
    right_foot = limbs["right_foot"]

    prev_left_foot = prev_limbs["left_foot"]
    prev_right_foot = prev_limbs["right_foot"]

    left_dist = distance_limbs(left_foot, prev_left_foot)
    right_dist = distance_limbs(right_foot, prev_right_foot)

    return left_dist/time_diff, right_dist/time_diff  

# Compute center of gravity
def center_of_gravity(pose_landmarks):
    # Limbs
    limbs = list(get_limbs(pose_landmarks).values())
    print(limbs)
    # Weights of each limb
    # Approximation of the weight of each limb
    # https://bionumbers.hms.harvard.edu/bionumber.aspx?id=109721
    left_hand_weight = 1
    right_hand_weight = 1
    left_elbow_weight = 1
    right_elbow_weight = 1
    left_foot_weight = 0.75
    right_foot_weight = 0.75
    left_shoulder_weight = 24
    right_shoulder_weight = 24
    left_knee_weight = 2.2
    right_knee_weight = 2.2
    left_hip_weight = 40
    right_hip_weight = 40
    head_weight = 12.4
    weights = [left_hand_weight, right_hand_weight, left_elbow_weight, right_elbow_weight, left_foot_weight, right_foot_weight, left_shoulder_weight, right_shoulder_weight, left_knee_weight, right_knee_weight, left_hip_weight, right_hip_weight, head_weight]
    # Get the x and y coordinates of the pose landmarks
    x = [limb.x*w for limb, w in zip(limbs, weights)]
    y = [limb.y*w for limb, w in zip(limbs, weights)]

    # Compute the center of gravity
    sum_weights = sum(weights)
    x_center = sum(x)/sum_weights
    y_center = sum(y)/sum_weights

    return x_center, y_center

def draw_center_of_gravity(frame, pose_landmarks):
    x_center, y_center = center_of_gravity(pose_landmarks)
    cv2.circle(frame, (int(x_center*frame.shape[1]), int(y_center*frame.shape[0])), 5, (255, 255, 0), -1)


# Draw the tension of each limb (red arrows)
def draw_tension(frame, pose_landmarks):
    def get_tension(a, b, len=1/2):
        return (1 + len)*b[0] - len*a[0], (1 + len)*b[1] - len*a[1]

    # get COG
    x_center, y_center = center_of_gravity(pose_landmarks)
    # get the limbs
    limbs = get_limbs(pose_landmarks)
    left_hand = limbs["left_hand"]
    right_hand = limbs["right_hand"]
    left_foot = limbs["left_foot"]
    right_foot = limbs["right_foot"]
    
    # compute tensions of each limb depending on the distance and angle
    tension_left_hand = get_tension((x_center, y_center), (left_hand.x, left_hand.y))
    tension_right_hand = get_tension((x_center, y_center), (right_hand.x, right_hand.y))
    tension_left_foot = get_tension((x_center, y_center), (left_foot.x, left_foot.y))
    tension_right_foot = get_tension((x_center, y_center), (right_foot.x, right_foot.y))

    # draw the tensions on the frame
    cv2.arrowedLine(frame, (int(left_hand.x*frame.shape[1]), int(left_hand.y*frame.shape[0])), (int(tension_left_hand[0]*frame.shape[1]), int(tension_left_hand[1]*frame.shape[0])), (255, 0, 0), 2)
    cv2.arrowedLine(frame, (int(right_hand.x*frame.shape[1]), int(right_hand.y*frame.shape[0])), (int(tension_right_hand[0]*frame.shape[1]), int(tension_right_hand[1]*frame.shape[0])), (255, 0, 0), 2)
    cv2.arrowedLine(frame, (int(left_foot.x*frame.shape[1]), int(left_foot.y*frame.shape[0])), (int(tension_left_foot[0]*frame.shape[1]), int(tension_left_foot[1]*frame.shape[0])), (255, 0, 0), 2)
    cv2.arrowedLine(frame, (int(right_foot.x*frame.shape[1]), int(right_foot.y*frame.shape[0])), (int(tension_right_foot[0]*frame.shape[1]), int(tension_right_foot[1]*frame.shape[0])), (255, 0, 0), 2)

# Compute the angle between 3 points
def compute_angle(a,b,c):
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(cosine_angle)
    
    return angle

# Compute the angle between 3 limbs
def compute_angle_limbs(limb1, limb2, limb3):
    a = np.array([limb1.x, limb1.y])
    b = np.array([limb2.x, limb2.y])
    c = np.array([limb3.x, limb3.y])

    return np.degrees(compute_angle(a, b, c))

# Compute angles between different limbs
def get_angles(pose_landmarks):
    # Get the limbs
    limbs = get_limbs(pose_landmarks)
    left_hand = limbs["left_hand"]
    right_hand = limbs["right_hand"]
    left_elbow = limbs["left_elbow"]
    right_elbow = limbs["right_elbow"]
    left_foot = limbs["left_foot"]
    right_foot = limbs["right_foot"]
    left_shoulder = limbs["left_shoulder"]
    right_shoulder = limbs["right_shoulder"]
    left_knee = limbs["left_knee"]
    right_knee = limbs["right_knee"]
    left_hip = limbs["left_hip"]
    right_hip = limbs["right_hip"]
    
    # Compute the angles
    angles = []
    angles.append(compute_angle_limbs(left_shoulder, left_elbow, left_hand))
    angles.append(compute_angle_limbs(right_shoulder, right_elbow, right_hand))
    angles.append(compute_angle_limbs(left_hip, left_knee, left_foot))
    angles.append(compute_angle_limbs(right_hip, right_knee, right_foot))
    angles.append(compute_angle_limbs(left_shoulder, left_hip, left_knee))
    angles.append(compute_angle_limbs(right_shoulder, right_hip, right_knee))
    angles.append(compute_angle_limbs(left_elbow, left_shoulder, left_hip))
    angles.append(compute_angle_limbs(right_elbow, right_shoulder, right_hip))
    return angles


# ----------- Ideas --------------
# Tell if the climber failed or succeeded the route
# # Shows the center of gravity of the climber
# # Shows the tension of each limb

# How to know when the climbers are climbing? 
    # Pose dynamics: identify patterns in the pose landmarks that shows the climber is climbing/ is starting/ has finished climbing
    # Detection: Detect the starting and ending holds of the climb
    # Compute pace and speed of the climber

# https://www.mdpi.com/2076-3417/13/4/2700
# Can call once getLimbs() and add it to the parameters of the functions
# Otsu's method for thresholding the image
# Add argument parser for the different options (draw landmarks, draw tension, draw center of gravity, etc.)
# --------------------------------
    
if __name__ == "__main__":
    detector = create_detector()
    run_pose_estimation(detector, "pose_estimation/climb_video.mp4")
