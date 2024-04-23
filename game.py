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
        # Load game elements
        self.score = 0

        # Initialize the enemies
        self.green_enemy = Enemy(GREEN)
        self.red_enemy = Enemy(RED)

        # Version of the game
        self.level = 0

        # Create the hand detector
        self.detector = mp.solutions.hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        # Load video
        self.video = cv2.VideoCapture(0)

        # Initialize the maze
        self.maze = Maze()


    def draw_maze(self, image):
        """
        Draws the maze onto the image.
        Args:
            image (Image): The image to draw on
        """
        # Draw the maze walls on the image
        for wall in self.maze.walls:
            cv2.line(image, wall[0], wall[1], BLUE, 2)

    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # Initalize the time 
        self.start_time = time.time()
        self.test_time = time.time()
        # Ask which version
        self.version_type = input("Which version are you playing in (0 for regular, 1 for timed, 2 for infinite spawning): ")
        if self.version_type == "1":
            self.level = 1
        elif self.version_type == "2":
            self.level = 2
        


        # Run until we close the video  
        while self.video.isOpened():
            # Get the current frame
            frame = self.video.read()[1]
            self.set_time = time.time()
            self.current_time = self.set_time - self.test_time
            if self.current_time > 5.1:
                self.test_time = self.set_time
            print(self.current_time)

            # Convert it to an RGB image
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # The image comes mirrored - flip it 
            image = cv2.flip(image, 1)

            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)

            # Draw the maze on the image 
            self.draw_maze(image)
            
            # Draw the enemy on the image 
            self.green_enemy.draw(image)
            self.red_enemy.draw(image)
            
            # Draw the score on the image 
            cv2.putText(image, 
                        str(self.score),
                        (50, 50),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=GREEN,
                        thickness=2)
            
            # "Time-it" option (displays time in console after user gets a score of 10)
            if self.score == 10 and self.level == 1:
                self.video.release()
                cv2.destroyAllWindows()
                self.end_time = time.time()
                self.time_elapsed = self.end_time - self.start_time
                print("Time elapsed:", self.time_elapsed)

            # Infinite spawning option
            if self.level == 2: 
                if self.current_time >= 5.0 and self.current_time <= 5.1:
                    print("WORKS")
                    self.other_green = Enemy(GREEN)
                    self.other_red = Enemy(RED)
                    self.other_green.draw(image)
                    self.other_red.draw(image)

            # Draw the hand landmarks
            # self.draw_landmarks_on_hand(image, results)
            self.check_enemy_kill(image, results)

            # Change the color of the frame back
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imshow('Hand Tracking', image)

            # Break the loop if the user presses 'q'
            if cv2.waitKey(50) & 0xFF == ord('q'):
                print(self.score)
                break
        
        # Release our video and close all windows 
        self.video.release()
        cv2.destroyAllWindows()

class Maze:
    def __init__(self):
        # Define the maze walls
        self.walls = [
            ((50, 50), (550, 50)),  
            ((50, 50), (50, 350)),  
            ((550, 50), (550, 350)), 
            ((50, 350), (550, 350)),  
            ((250, 50), (250, 350)),  
            ((450, 50), (450, 350)), 
        ]


if __name__ == "__main__":        
    g = Game()
    g.run()