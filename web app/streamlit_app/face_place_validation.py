from imports import *
# Load .env vars
load_dotenv()	

import cv2 as cv2
from PIL import Image

def create_gs_user_photos_ds(user_id):
    """Create a dataset of user face images using real time capturing

    Parameters
    ----------
    user_id: int
        The id of the user we are creating the face ds for

    Returns
    -------
    None
    """
    
    gs_ds_dir = Path("imgs") / "user_photos" / "gs"
    
    # Remove previous ds if exists and create a new empty one.
    try:
        shutil.rmtree(gs_ds_dir)
    except:
        pass
    os.mkdir(gs_ds_dir)
    
    try:
    
        cam = cv2.VideoCapture(0)
        cam.set(3, 640) # set video width
        cam.set(4, 480) # set video height
        face_detector = cv2.CascadeClassifier(str(Path('cv2_Cascades') / 'haarcascade_frontalface_default.xml'))

        print("[INFO] Initializing face capture. Look the camera and wait. Press 's' to save a frame and 'ESC' to stop...")
        # Initialize individual sampling face count
        count = 0
        while(True):
            ret, img = cam.read()
            #img = cv2.flip(img, -1) # flip video image vertically
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
                
            cv2.imshow('image', img)

            k = cv2.waitKey(100) & 0xff # Press 'ESC' for exiting video 
            if k == ord('s'): # press "s" to save face from current frame
                count += 1
                # Save the captured image into the datasets folder
                cv2.imwrite(Path(gs_ds_dir) / "User.{0}.{1}.jpg".format(user_id, count), gray[y:y+h,x:x+w])
                
            elif k == 27: # if ESC is pressed
                break
            elif count >= 30: # Take 30 face sample and stop video
                 break
    except Exception as ex:
        print("[Error]:", ex)
        
    finally:
        # Do a bit of cleanup
        print("\n [INFO] Exiting Program and cleanup stuff")
        cam.release()
        cv2.destroyAllWindows()
    
    return None


def face_rec_train(face_gs_imgs_dir):
    """Train CV2 classifier on face imgs dataset

    Parameters
    ----------
    face_gs_imgs_dir: str
        The path to the dir containing face imgs ds

    Returns
    -------
    None
    """
    
    # Path for face image database
    path = face_gs_imgs_dir
    # Use LOCAL BINARY PATTERNS HISTOGRAMS for face recognition by creating a representation of the texture pattern in the neighberhood 
    # to read more: https://www.geeksforgeeks.org/face-recognition-with-local-binary-patterns-lbps-and-opencv/
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    detector = cv2.CascadeClassifier(str(Path('cv2_Cascades') / 'haarcascade_frontalface_default.xml'))
    
    # function to get the images and label data
    def getImagesAndLabels(path):
        imagePaths = [f for f in path.glob("*.jpg")]     
        faceSamples=[]
        ids = []
        for imagePath in imagePaths:
            print(str(path))
            print(str(imagePath))
            PIL_img = Image.open(imagePath).convert('L') # grayscale
            img_numpy = np.array(PIL_img,'uint8')
            id = int(os.path.split(imagePath)[-1].split(".")[1]) # filename has patter: User.[user_id].[photo_num].jpg
            faces = detector.detectMultiScale(img_numpy)
            for (x,y,w,h) in faces:
                faceSamples.append(img_numpy[y:y+h,x:x+w])
                ids.append(id)
        return faceSamples,ids
    
    print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
    faces,ids = getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    # Save the model into trainer/trainer.yml
    recognizer.write(str(Path('face_trainer') / 'trainer.yml'))
    # Print the numer of faces trained and end program
    print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))
    
    return None


def validate_face_in_img(captured_img_path, val_usr_name, plot=False):
    """Validates User's face in a saved photo using our trained CV2 classifier

    Parameters
    ----------
    captured_img_path: str
        The path to the saved captured image for the user in front of validation place.
    val_usr_name: str
        The name of the user we are validating in the captured image
    plot: bool, optional
        A flag to set if we want to plot face validation output image

    Returns
    -------
    user_validated: bool
        Wether the user is validated
    msg: str:
        The validation output msg
    """
    
    # Convert pathlib.Path to str
    img_path = str(captured_img_path)
    # Debugging
    #print("> Image:", img_path)

    user_validated = False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    trainer_path = Path('face_trainer') / 'trainer.yml'
    recognizer.read(str(trainer_path))
    faceCascade = cv2.CascadeClassifier(str(Path('cv2_Cascades') / 'haarcascade_frontalface_default.xml'))
    font = cv2.FONT_HERSHEY_SIMPLEX
    #iniciate id counter
    id = 0
    # names related to ids: example ==> Omar: id=1,  etc
    names = ['User_0', 'Omar']
    
    # read img from path and convert it into Numpy array in gray scale
    img = cv2.imread(img_path)
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
        minSize = (15, 15),
       )
    
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 4)
        id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # If confidence is less them 100 ==> "0" : perfect match 
        if (confidence < 100):
            user_name = names[id]
            if user_name == val_usr_name:
                confidence = "  {0}%".format(round(100 - confidence))
                user_validated = True
            else:
                id = "unknown"
                confidence = "  {0}%".format(round(100 - confidence))

        else:
            id = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
                    img, 
                    str(id), 
                    (x+5,y-5), 
                    font, 
                    1, 
                    (255,255,255), 
                    2
                   )
        cv2.putText(
                    img, 
                    str(confidence), 
                    (x+5,y+h-5), 
                    font, 
                    1, 
                    (255,255,0), 
                    1
                   )  
        
    if user_validated:
        msg = "\t- Good Job! {0}'s face validated!".format(val_usr_name)
    else:
        msg = "\t- Can't find {0} in the image".format(val_usr_name)
        
    if plot:
        # Change BGR to RGB for plotting
        img = img[...,::-1]
        # Show validation rectangle and label
        plt.imshow(img)
        plt.show()

    return user_validated, msg


# Tutorial from OpenCV: https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html
def check_bg_similarity(captured_img_path, plc_id, plot = False):
    """Validates User's face in a saved photo using our trained CV2 classifier

    Parameters
    ----------
    captured_img_path: str
        The path to the saved captured image for the user in front of validation place.
    plc_id: str
        The id of the place (Google Place) with which we want to check image similarity
    plot: bool, optional
        A flag to set if we want to plot background similarity output image

    Returns
    -------
    bg_similar: bool
        Wether the captured image and the Google Places image for the selected place are similar enough
    msg: str:
        The similarity check output msg
    """
    
    # Convert pathlib.Path to str
    captured_img_path = str(captured_img_path)
    # Debugging
    #print("> Image:", captured_img_path)
    
    bg_similar = False
    msg = ""
    
    MAX_DISTANCE_THRESH = 0.70
    MIN_MATCH_COUNT = 10
    
    # Read visited place img, the one we have saved from Google Places while choosing today's plan in grayscale
    plc_img_path = str(Path("imgs") / "plc_{0}.jpg".format(plc_id))
    img_plc = cv2.imread(plc_img_path, cv2.IMREAD_GRAYSCALE) # queryImage
    
    # Read captured img in gray scale
    img_cap = cv2.imread(captured_img_path, cv2.IMREAD_GRAYSCALE) # trainImage
    img_cap = cv2.flip(img_cap, 1) # flip horizontally to original
    
    # Scale img_cap (usually much larger than img_plc)
    img_plc_h, img_plc_w = img_plc.shape
    img_cap_h, img_cap_w = img_cap.shape
    
    # Calculate scaling factors
    h_s_f = round(img_plc_h / img_cap_h, 2)
    w_s_f = round(img_plc_w / img_cap_w, 2)
    
    img_cap = cv2.resize(img_cap, (0,0), fx=h_s_f, fy=w_s_f) 
    
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    
    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_plc, None)
    kp2, des2 = sift.detectAndCompute(img_cap, None)
    
    # Set algorithm parameters
    FLANN_INDEX_KDTREE = 1
    N_TREES = 5
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = N_TREES)
    search_params = dict(checks = 50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    
    # Find matches
    matches = flann.knnMatch(des1, des2, k=2) # matching between (2) points
    
    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < (MAX_DISTANCE_THRESH * n.distance):
            good_matches.append(m)
    
    enough_matches = len(good_matches) >= MIN_MATCH_COUNT
    
    if plot:

        # If enough matches are found, we extract the locations of matched keypoints in both the images.
        # Then, we pass them to find the perspective transformation. Once we get this 3x3 transformation 
        # matrix, we use it to transform the corners of queryImage to corresponding points in trainImage then
        # we draw it.
        
        try:
            
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img_plc.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            img_cap = cv2.polylines(img_cap,[np.int32(dst)],True,255,3, cv2.LINE_AA)

            # Finally we draw our inliers (if successfully found the object) or matching keypoints (if failed).
            draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                singlePointColor = None,
                matchesMask = matchesMask, # draw only inliers
                flags = 2
            )
        except:
            # Init as empty dict
            draw_params = dict()
        
        img_marks = cv2.drawMatches(img_plc, kp1, img_cap, kp2, good_matches, None, **draw_params)
        plt.imshow(img_marks, 'gray'),plt.show()
        
    # Set a condition that at least MIN_MATCH_COUNT matches are to be there to find the object. Otherwise simply show a message saying not enough matches are present.
    if enough_matches:
        bg_similar = True
        msg = "Place validated - {0} good landmark matches.".format(len(good_matches))
    else:
        bg_similar = False
        msg = "Not enough landmark matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT)

    return bg_similar, msg


def validate_usr_in_plc(captured_img_path, val_usr_name, plc_id):
    """Validates User's face in a saved photo captured in a specific place using our trained CV2 classifier

    Parameters
    ----------
    captured_img_path: str
        The path to the saved captured image for the user in front of validation place
    val_usr_name: str
        The name of user being validated. This will be used by face classifier
    plc_id: str
        The id of the place (Google Place) with which we want to check background similarity

    Returns
    -------
    validated: bool
        Wether validation is successful
    msg: str
        The face in place validation output msg
    """
    
    validated = False
    msg = ""
    
    # 1- Validated user in pic using our OpenCV2 trained model
    user_face_validated, m = validate_face_in_img(captured_img_path, val_usr_name)
    if not user_face_validated:
        msg = "Can't recognise {0}'s face. Please make sure the face is not too small and that it doesn't appear cropped in the captured image.".format(val_usr_name)
        return validated, msg
    
    # 2- Check if the captured image has a background that is similar to the place image provided by Google Places
    bg_similar, m = check_bg_similarity(captured_img_path, plc_id)
    if not bg_similar:
        msg = "The captured photo doesn't seem similar to the selected photo from Google Places for this place. Can you try showing more landmarks in the taken photo?"
        return validated, msg
    
    validated = True
    msg = "Great job! Validated"
    
    return validated, msg

