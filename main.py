import math
import cv2
import numpy as np
import imutils

videocap = cv2.VideoCapture('Microrobot.mp4')

while videocap.isOpened():
    success, frame = videocap.read()
    if success:

        # Pre-process image
        # cv2.imshow('RAW', frame)
        grayScaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # dilate = cv2.dilate(grayScaled, None, iterations=4)
        # gaussian = cv2.GaussianBlur(dilate, (5, 5), cv2.BORDER_DEFAULT)
        # cv2.imshow("dilate", gaussian)
        darkMask = cv2.inRange(grayScaled, 0, 60)  # robot
        darkMask2 = cv2.erode(darkMask, None, iterations=2)
        lightMask = cv2.inRange(grayScaled, 80, 230)  # cargo
        lightMask2 = cv2.erode(lightMask, None, iterations=2)
        #cv2.imshow('Light Mask', lightMask2)
        #cv2.imshow('Dark Mask', darkMask)
        cnts_rbt = cv2.findContours(darkMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        cnts_cargo = cv2.findContours(lightMask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c_cargo = sorted(cnts_cargo, key=cv2.contourArea, reverse=True)[:5]

        # Only proceed if at least one contour was found
        if len(cnts_rbt) > 0:
            c_rbt = sorted(cnts_rbt, key=cv2.contourArea, reverse=True)[:5]  # get the 5 largest contour

            # Detect Robot---------------------------------------------------------------------------------------
            ((rx1, ry1), (w, l), angle1) = cv2.minAreaRect(c_rbt[0])
            box = cv2.boxPoints(((rx1, ry1), (w, l), angle1))  # cv2.boxPoints(rect) for OpenCV 3.x
            box = np.int0(box)
            cv2.drawContours(frame, c_rbt, 0, (0, 0, 255), 2)
            M_rbt = cv2.moments(c_rbt[0])
            center_rbt = (int(M_rbt["m10"] / M_rbt["m00"]), int(M_rbt["m01"] / M_rbt["m00"]))

            # Draw robot contours and display real-time positions
            cv2.putText(frame, 'Robot', center_rbt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            # cv2.drawContours(frame, [box], 0, (0, 255, 0), 2)
            cv2.putText(frame, 'Robot X: %f' % rx1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.putText(frame, 'Robot Y: %f' % ry1, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

            # Detect cargo
            ((x, y), radius) = cv2.minEnclosingCircle(c_cargo[0])

            # Draw cargo contours and display real-time positions
            cv2.putText(frame, 'Cargo', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            cv2.putText(frame, 'Cargo', (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.putText(frame, 'Cargo X: %f' % x, (380, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.putText(frame, 'Cargo Y: %f' % y, (380, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            print('Real-Time Position of Cargo: X: %f, Y: %f' % (x, y))

            # Calculate robot orientation
            rx2 = int(rx1)
            ry2 = int(ry1)
            dx = center_rbt[0] - rx2
            dy = center_rbt[1] - ry2
            rx3 = rx2 + 15 * dx
            ry3 = ry2 + 15 * dy
            if not dx == 0:
                angle2 = math.degrees(np.arctan(dy / dx))
                if dx < 0 and dy < 0:
                    angle2 += 270
                if dx > 0 and dy < 0:
                    angle2 += 90
                if dx < 0 and dy > 0:
                    angle2 += 270
                if dx > 0 and dy > 0:
                    angle2 += 0
            elif dy < 0:
                angle2 = 360
            elif dx == 0 and dy > 0:
                angle2 = 180
            if dy == 0:
                angle2 = 270

            # Calculate Angle to Cargo
            angleToCargo = math.degrees(np.arctan(abs(y - ry2) / abs(x - rx2))) + 270
            if angle2 < 90:
                d_angle = angleToCargo - angle2 - 360
            else:
                d_angle = angleToCargo - angle2

            # Display robot orientation and angle to cargo
            cv2.putText(frame, 'Angle to Cargo: %f' % angleToCargo, (380, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            cv2.putText(frame, 'Delta_Angle: %f' % (d_angle-10), (380, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)
            print('Real-Time Position of Robot: X: %f, Y: %f, Angle: %f ' % (rx1, ry1-5, angle2))
            cv2.putText(frame, 'Robot Orientation: %f' % angle2, (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

            # Draw arrow between robot and cargo to show cargo's position in relative to robot
            cv2.arrowedLine(frame,
                            (rx2, ry2),
                            (int(x), int(y)), (255, 0, 0), 3)

        # Use contour size to determine if cargo is found
        mediumMask = cv2.inRange(grayScaled, 10, 250)
        cnts_total = cv2.findContours(mediumMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        c_total = sorted(cnts_total, key=cv2.contourArea, reverse=True)[:5]
        area = cv2.contourArea(c_total[0])
        if area > 5500:
            cv2.putText(frame, 'Cargo Found', (30, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (50, 180, 120), 3)
            cargo_found = True
        else:
            cv2.putText(frame, 'Looking for Cargo', (30, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (35, 70, 100), 3)

        # Use distance to determine if cargo is picked up
        distance = np.sqrt(abs(int(x) - int(rx2)) ** 2 + abs(int(y) - int(ry2)) ** 2)
        if distance < 40:
            cv2.putText(frame, 'Cargo Picked up', (350, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (35, 100, 150), 3)
            cargo_picked_up = True

        # Display distance to cargo
        cv2.putText(frame, 'Distance to Cargo: %f' % (distance-33), (380, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 2)

        # Draw coordinate system
        c_x1 = int(rx1)
        c_y1 = int(ry1)
        length = 80
        height = 80
        cv2.arrowedLine(frame, (c_x1, c_y1), (c_x1 + length, c_y1), (220, 170, 0), 2)
        cv2.arrowedLine(frame, (c_x1, c_y1), (c_x1, c_y1 + height), (220, 170, 0), 2)
        cv2.putText(frame, 'X', (c_x1 + length + 5, c_y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 100, 150), 2)
        cv2.putText(frame, 'Y', (c_x1, c_y1 + height + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (35, 100, 150), 2)

        # Show continuous frame with information
        window_vertical = np.vstack((grayScaled, darkMask,lightMask2))
        cv2.imshow('Stacked Windows', window_vertical)
        cv2.imshow('Frame', frame)
        cv2.moveWindow('Frame', 340, -100)

    # Press 'space to pause, ''q' to quit
    key = cv2.waitKey(1)
    if key == 32:
        cv2.waitKey()
    elif key == ord('q'):
        break
