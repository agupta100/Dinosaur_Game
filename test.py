"""
A game that uses hand tracking to 
hit and destroy green circle enemies.

@author: Nandhini Namasivayam
@version: March 2024

edited from: https://i-know-python.com/computer-vision-game-using-mediapipe-and-python/
"""

import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import cv2
import random
import time

RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# Library Constants
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkPoints = mp.solutions.hands.HandLandmark
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode
DrawingUtil = mp.solutions.drawing_utils

class Enemy:
    """
    A class to represent a random circle
    enemy. It spawns randomly within 
    the given bounds.
    """
    def __init__(self, color, screen_width=600, screen_height=400):
        self.color = color
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.respawn()
    
    def respawn(self):
        """
        Selects a random location on the screen to respawn
        """
        self.x = random.randint(50, self.screen_width)
        self.y = random.randint(50, self.screen_height)
    
    def draw(self, image):
        """
        Enemy is drawn as a circle onto the image

        Args:
            image (Image): The image to draw the enemy onto
        """
        cv2.circle(image, (self.x, self.y), 25, self.color, 5)
      
class Game:
    def __init__(self):
        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)

        # Load video
        self.video = cv2.VideoCapture(0)

        hand_landmarks_list = detection_result.hand_landmarks

         # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Get the coordinates of just the index finger (the tip of the index finger)
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]


    def draw_landmarks_on_hand(self, image, detection_result):
        """
        Draws all the landmarks on the hand
        Args:
            image (Image): Image to draw on
            detection_result (HandLandmarkerResult): HandLandmarker detection results
        """
        # Get a list of the landmarks
        hand_landmarks_list = detection_result.hand_landmarks
        
        # Loop through the detected hands to visualize.
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]

            # Save the landmarks into a NormalizedLandmarkList
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])

            # Draw the landmarks on the hand
            DrawingUtil.draw_landmarks(image,
                                       hand_landmarks_proto,
                                       solutions.hands.HAND_CONNECTIONS,
                                       solutions.drawing_styles.get_default_hand_landmarks_style(),
                                       solutions.drawing_styles.get_default_hand_connections_style())
            

    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # Run until we close the video  
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes mirrored - flip it 
            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            self.draw_landmarks_on_hand(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2.imshow('Hand Tracking', image)

            # Get image details
            imageHeight, imageWidth = image.shape[:2]

            # Get a list of the landmarks
            hand_landmarks_list = detection_result.hand_landmarks
        
            # Loop through the detected hands to visualize.
            for idx in range(len(hand_landmarks_list)):
                hand_landmarks = hand_landmarks_list[idx]

            # Get the coordinates of just the index finger (the tip of the index finger)
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]

            pixelCoordinates = DrawingUtil._normalized_to_pixel_coordinates(finger.x,
                                                                            finger.y,
                                                                            imageWidth, 
                                                                            imageHeight)
            if pixelCoordinates:
                # Draw the circle around the index finger 
                cv2.circle(image,
                        (pixelCoordinates[0], pixelCoordinates[1]),
                        25,
                        GREEN,
                        5)
                # Check if we intercept the enemy
                self.check_enemy_intercept(pixelCoordinates[0],
                                           pixelCoordinates[1],
                                           self.green_enemy,
                                           image)
                


            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        
        # Release our video and close all windows 
        self.video.release()
        cv2.destroyAllWindows()

        
if __name__ == "__main__":        
    g = Game()
    g.run()