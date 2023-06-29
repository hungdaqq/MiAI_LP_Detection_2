import cv2
from glob import glob
from lib_detection import load_model, detect_lp, im2single
from os.path import splitext, basename

input_dir  = '/home/hung/Downloads/GreenParking/'
output_dir = '/home/hung/Downloads/data/greenparking/'

# cap = cv2.VideoCapture('/home/hung/Downloads/test.MOV')

# if (cap.isOpened()== False): 
#   print("Error opening video stream or file")


wpod_net_path = 'wpod-net_update1.json'
wpod_net = load_model(wpod_net_path)

# while(cap.isOpened()):
#     ret, Ivehicle = cap.read()
#     if ret == True:
#         cv2.imshow('Frame',Ivehicle)
#         # Ivehicle = cv2.imread(frame)
#         ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
#         side  = int(ratio*288.)
#         bound_dim = min(side + (side%(2**4)),608)
#         Llp, LpImgs, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
#         if (len(LpImgs)):
#             Ilp = LpImgs[0]
#             print(Ilp.shape)
#             Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
#             Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)
#         if cv2.waitKey(25) & 0xFF == ord('q'):
#             break
#     else: 
#         break
 
# cap.release()
# cv2.destroyAllWindows()

imgs_paths = glob('%s/*.jpg' % input_dir)
print('Searching for license plates using WPOD-NET')

for i,img_path in enumerate(imgs_paths):
    bname = splitext(basename(img_path))[0]
    Ivehicle = cv2.imread(img_path)
    ratio = float(max(Ivehicle.shape[:2]))/min(Ivehicle.shape[:2])
    side  = int(ratio*288.)
    bound_dim = min(side + (side%(2**4)),608)
    Llp , LpImgs, lp_type = detect_lp(wpod_net, im2single(Ivehicle), bound_dim, lp_threshold=0.5)
    if (len(LpImgs)):
        Ilp = LpImgs[0]
        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_BGR2GRAY)
        # Ilp = cv2.cvtColor(Ilp, cv2.COLOR_GRAY2BGR)

        cv2.imwrite('%s/%s_lp.png' % (output_dir,bname),Ilp*255.)