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

      
class Game:
    def __init__(self):
        # Load game elements
        self.score = 0
        self.somethingelse = "OK"

        # Version of the game
        self.level = 0

        # Create the hand detector
        base_options = BaseOptions(model_asset_path='data/hand_landmarker.task')
        options = HandLandmarkerOptions(base_options=base_options,
                                                num_hands=2)
        self.detector = HandLandmarker.create_from_options(options)
        self.dinoimage = cv2.imread("dinosaur.png", -1)
        self.dinoimageHeight, self.dinoimageWidth = self.dinoimage.shape[:2]
        self.listofcacti = []
        colorlist = [RED, GREEN, BLUE]
        for i in range(20):
            rando = random.randint(0,2)
            color = colorlist[rando]
            cacti = Cacti(color, 300, 200)
            self.listofcacti.append(cacti)






        # Load video
        self.video = cv2.VideoCapture(0)

        
            
    
    
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
            finger = hand_landmarks[HandLandmarkPoints.INDEX_FINGER_TIP.value]
            pixelCoord = DrawingUtil._normalized_to_pixel_coordinates(finger.x,
                                                                      finger.y,
                                                                      self.imageWidth,
                                                                      self.imageHeight)
            if pixelCoord:
                print("Dino: " , self.dinoimageHeight , ", " , self.dinoimageWidth)
                print("Image: " , (pixelCoord[0]-self.dinoimageHeight//2 - pixelCoord[0]-self.dinoimageHeight//2) , ", " , (pixelCoord[1]-self.dinoimageWidth//2 - pixelCoord[1]-self.dinoimageWidth//2))
                cv2.circle(image,
                           (pixelCoord[0],pixelCoord[1]),
                           25,
                           RED,
                           5)
                return [pixelCoord[0], pixelCoord[1]]
        return None
        
    def check_enemy_intercept(self, finger_x, finger_y, enemy):
        if (enemy.getx() <= finger_x + 50 and enemy.getx() >= finger_x - 50) and (enemy.gety() <= finger_y + 50 and enemy.gety() >= finger_y - 50):
            return "GAME OVER!"
            
        
    def run(self):
        """
        Main game loop. Runs until the 
        user presses "q".
        """    
        # Initalize the time 
        self.start_time = time.time()
        self.test_time = time.time()

    


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
            self.imageHeight, self.imageWidth = image.shape[:2]


            # Convert the image to a readable format and find the hands
            to_detect = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            results = self.detector.detect(to_detect)
            if (self.draw_landmarks_on_hand(image, results) != None):
                somethinghere = self.draw_landmarks_on_hand(image, results)
                for i in self.listofcacti:
                    self.somethingelse = self.check_enemy_intercept(somethinghere[0], somethinghere[1], i)
                    i.random_pattern()
                    i.drawCircle(image)
            else:
                self.draw_landmarks_on_hand(image, results)
            if self.somethingelse == "GAME OVER!":
                break


    
            
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

        

class Cacti:
    def __init__(self, color, x, y, screen_width=800, screen_height=600):
        self.x = x
        self.y = y 
        
        self.speed = 9
        self.move_to_x = x
        self.move_to_y = y
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.color = color
        


    def getCoords(self):
        return (self.x, self.y)
    
    def getx(self):
        return self.x
    def gety(self):
        return self.y
    
    def updatePosition(self):
        self.x += random.randint(0, 10)
        self.y += random.randint(0, 10)
        if self.x >= self.screen_width - 5:
            self.x = self.screen_width - 5
        if self.x <= 5:
            self.x = 5
        if self.y >= 5:
            self.y = 5
        if self.y <= self.screen_width - 5:
            self.y = self.screen_width - 5
        

    
    def random_pattern(self):
        if (self.x+10 > self.move_to_x and self.x-10 < self.move_to_x) and (self.y+10 > self.move_to_y and self.y-10 < self.move_to_y):
            self.move_to_x = random.randint(0, self.screen_width - 50)
            self.move_to_y = random.randint(0, self.screen_height - 50)
        
        if not (self.x+self.speed > self.move_to_x and self.x-self.speed < self.move_to_x): 
            if self.move_to_x > self.x:
                self.x += self.speed
            elif self.move_to_x < self.x:
                self.x -= self.speed
        if not (self.y+self.speed > self.move_to_y and self.y-self.speed < self.move_to_y): 
            if self.move_to_y > self.y:
                self.y += self.speed
            elif self.move_to_y < self.y:
                self.y -= self.speed
    
    def drawCircle(self, image):
        cv2.circle(image,
                           (self.x,self.y),
                           25,
                           self.color,
                           5)
    
    def draw(self, image):
        cv2.rectangle(image, (self.x - self.size // 2, self.y - self.size // 2), (self.x + self.size // 2, self.y + self.size // 2), self.color, -1)
 

        
if __name__ == "__main__":        
    g = Game()
    g.run()