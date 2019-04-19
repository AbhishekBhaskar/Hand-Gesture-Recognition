import handy
import cv2
import math
from Hand import Hand
import pyautogui

# getting video feed from webcam
cap = cv2.VideoCapture(0)

# capture the hand histogram by placing your hand in the box shown and
# press 'A' to confirm
# source is set to inbuilt webcam by default. Pass source=1 to use an
# external camera.
hist = handy.capture_histogram(source=0)

while True:
    ret, frame = cap.read()
    #print(frame)
    if not ret:
        break

    # to block a faces in the video stream, set block=True.
    # if you just want to detect the faces, set block=False
    # if you do not want to do anything with faces, remove this line
    handy.detect_face(frame, block=True)

    # detect the hand
    hand = handy.detect_hand(frame, hist)

    # to get the outline of the hand
    # min area of the hand to be detected = 10000 by default
    custom_outline = hand.draw_outline(
        min_area=10000, color=(0, 255, 255), thickness=2)



    # to get a quick outline of the hand
    quick_outline = hand.outline

    # draw fingertips on the outline of the hand, with radius 5 and color red,
    # filled in.
    for fingertip in hand.fingertips:
        cv2.circle(quick_outline, fingertip, 5, (0, 0, 255), -1)

    #angle = hand.checkAngle
    #for defect in hand.extractDefects:
    #   cv2.circle(quick_outline, defect, 5, (255, 0, 0), -1)
    l=0
    for defect in hand.getDefects:
        l += 1
        cv2.circle(quick_outline, defect, 5, (255, 0, 0), -1)
        #print("working")
    l+=1
    font = cv2.FONT_HERSHEY_SIMPLEX
    if l==1:
        cv2.putText(quick_outline, 'Zoom out', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        pyautogui.hotkey('command','-')
    elif l == 2:
        cv2.putText(quick_outline, 'Right', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        pyautogui.press('right')
    elif l==4:
        cv2.putText(quick_outline, 'Zoom in', (0, 50), font, 2, (0, 0, 255), 3, cv2.LINE_AA)
        pyautogui.hotkey('command', '+')

    #to get the centre of mass of the hand
    com = hand.get_center_of_mass()
    if com:
        cv2.circle(quick_outline, com, 10, (255, 0, 0), -1)

    cv2.imshow("Handy", quick_outline)

    # display the unprocessed, segmented hand
    # cv2.imshow("Handy", hand.masked)

    # display the binary version of the hand
    # cv2.imshow("Handy", hand.binary)

    k = cv2.waitKey(5)

    # Press 'q' to exit
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
