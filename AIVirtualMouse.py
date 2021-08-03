
import cv2
import numpy as np
import time
import autopy
import HandTracking as htm
####################
wCam, hCam = 640, 480
frameR = 100
Smoothening = 10
####################
pTime = 0
plocX, plocY = 0, 0
clocX, clocY = 0, 0
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)
detector = htm.handDetector(maxHands=1)
wScr, hScr = autopy.screen.size()
print(wScr, hScr)

while True:
    # 1、Find hand Landmarks
    success, img = cap.read()
    img = detector.findHands(img)
    lmList, bBox = detector.findPosition(img)
    # 2. Get the tip of the index and middle fingers
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        #print(x1, y1, x2, y2)
        # 3. Check which fingers are up
        fingers = detector.fingersUp()
        # print(fingers)
        cv2.rectangle(img, (frameR, frameR),
                      (wCam-frameR, hCam-frameR), (255, 0, 255), 2)
        # 4. OnLy Index Finger Moving Mode
        if fingers[1] == 1 and fingers[2] == 0:
            # 5. Convert Coordinates

            x3 = np.interp(x1, (frameR, wCam-frameR), (0, wScr))
            y3 = np.interp(y1, (frameR, hCam-frameR), (0, hScr))
            # 6. Smoothen values
            clocX = plocX+(x3-plocX)/Smoothening
            clocY = plocY+(y3-plocY)/Smoothening
            # 7. Move Mouse
            '''
            clocY = 10*clocY
            if clocY >= 768:
                clocY = 760
            if clocY <= 0:
                clocY = 10
            '''
            autopy.mouse.move(wScr - clocX, clocY)
            cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
            plocX, plocY = clocX, clocY
        # 8. Both Index and middle fingers are CLicking Mode
        if fingers[1] == 1 and fingers[2] == 1:
            # 9. Find distance between fingers
            length, img, lineInfo = detector.findDistance(8, 12, img)
            print(length)
            # 10. CLick mouse if distance short
            if length < 40:
                cv2.circle(img, (lineInfo[4], lineInfo[5]),
                           15, (0, 255, 0), cv2.FILLED)
                autopy.mouse.click()

        # 11. Frame Rate
    cTime = time.time()
    Fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img, "FPS="+str(int(Fps)), (20, 50),
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0))
    cv2.imshow("image", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("I'm done")
        break
