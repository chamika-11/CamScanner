import cv2
import numpy as np
import pytesseract
import os


#capture image          
def capture_image(save_path="camscanner.jpg"):

    cap = cv2.VideoCapture(0)  # Open the webcam

    if not cap.isOpened():  # Check if the webcam is opened
        print("Having an error with the camera!")
        return

    print("Press space to capture or ESC to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture the image")
            break

        cv2.imshow("Webcam feed", frame)  # Display live video stream

        key = cv2.waitKey(1)  # Wait for key press

        if key == 27:  # ESC key to exit
            print("Exiting...")
            break
        elif key == 32:  # Space bar to capture
            cv2.imwrite(save_path, frame)
            print(f"Image saved as {save_path}")
            break

    cap.release()  # Free up the webcam
    cv2.destroyAllWindows()  # Close all the OpenCV windows

    return save_path


#cropping the image
def detect_cropping(image_path, save_path="Editedimage.jpg"):
    
    image=cv2.imread(image_path)
    processed_image = preprocess_image(image_path)
    edgeCrop=cv2.Canny(processed_image,50,150)

    contours,_=cv2.findContours(edgeCrop,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)#Find the outlines,retrive only outmost contours
    contours=sorted(contours, key=cv2.contourArea, reverse=True)#largest file will be the document itself(sort contours by size)

    for contour in contours:
        peri=cv2.arcLength(contour,True)
        approx=cv2.approxPolyDP(contour,0.02*peri,True)#change the rectangle corners

        if len(approx)==4:
            x,y,w,h=cv2.boundingRect(approx)
            croped=image[y:y+h, x:x+w]

            cv2.imwrite(save_path,croped)
            print(f"document croped and save as{save_path}")
            return save_path
    
    print("No document detected")
    return None


#image proceccing
def preprocess_image(image_path):
    image=cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    sharpened = cv2.filter2D(thresh, -1, kernel)
    
    return sharpened


def deskew_image(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle

    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return rotated



#Extract text from the document
def Extract_text(image):
    text = pytesseract.image_to_string(image, config='--psm 6')
    return text.strip()


#save file
def save_file(text,save_dir="scanned_docs",filename="Docscantxt.txt"):
    
    #check if there already exists
    if not os.path.exists(save_dir):#check the path
        os.makedirs(save_dir)#if theres no file on the path
        file_path=os.path.join(save_dir,filename)
        
        with open (file_path,"w",encoding="utf-8") as file:
            file.write(text)
            print(f"file saved to {file_path}")



#full document scanning proceccing
def scan_document():
    captured_image = capture_image()
    if not captured_image:
        return
    
    cropped_image = detect_cropping(captured_image)
    if not cropped_image:
        return
    
    # Preprocess the cropped image with adaptive thresholding and deskewing
    preprocessed_image = preprocess_image(cropped_image)
    deskewed_image = deskew_image(preprocessed_image)
    
    # Pass the deskewed image (NumPy array) directly to Extract_text
    extract_text = Extract_text(deskewed_image)
    if extract_text:
        print(extract_text)
        save_file(extract_text)
    else:
        print("No text found")

        

if __name__ == "__main__":
    capture_image()
    scan_document()
