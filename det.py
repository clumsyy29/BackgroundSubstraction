import cv2

cap = cv2.VideoCapture('road.mp4')
fgbg = cv2.bgsegm.createBackgroundSubtractorMOG()

while True:
    ret, frame = cap.read()
    
    if not ret:
        break

    fgmask = fgbg.apply(frame)

    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, kernel)

    
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        
        area = cv2.contourArea(contour)
        
        
        if area < 100:
            continue

        
        x, y, w, h = cv2.boundingRect(contour)

        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

       
        aspect_ratio = float(w) / h

        
        cv2.putText(frame, f'Size: {area}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, f'Ratio: {aspect_ratio:.2f}', (x, y+h+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow('frame', frame)
    
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
