from Models.Unet import *
from libraries import *


if not torch.cuda.is_available():
    device=torch.device("cpu")
    print("Current device:", device)
else:
    device=torch.device("cuda")
    print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
    bf.get_memory()

CKP_DIR = "./Data/Models/Unet/checkpoint/"

model = UNet(n_channels=3, n_classes=7).to(device)
model.load_state_dict(torch.load(CKP_DIR+ "00050.pth"))
model.eval()

preprocess = bf.PREPROCESS()





def map_colors(mask):
    """
    colors = {"[0 0 0]" : [0,0,0],       #black   
              "[1 1 1]" : [255,255,102], #yellow
              "[2 2 2]" : [255, 0, 0],   #red
              "[3 3 3]" : [0,255,0],     #green
              "[4 4 4]" : [0,0,255],     #blue
              "[5 5 5]" : [255,0,255],   #pink
              "[6 6 6]" : [0, 255, 255]} #azzurro
    """
    mask[:,:,0][mask[:,:,0] == 1] = 255
    mask[:,:,1][mask[:,:,1] == 1] = 255
    mask[:,:,2][mask[:,:,2] == 1] = 255    

    mask[:,:,0][mask[:,:,0] == 2] = 255
    mask[:,:,1][mask[:,:,1] == 2] = 0
    mask[:,:,2][mask[:,:,2] == 2] = 0

    mask[:,:,0][mask[:,:,0] == 3] = 0
    mask[:,:,1][mask[:,:,1] == 3] = 255
    mask[:,:,2][mask[:,:,2] == 3] = 0

    mask[:,:,0][mask[:,:,0] == 4] = 0
    mask[:,:,1][mask[:,:,1] == 4] = 0
    mask[:,:,2][mask[:,:,2] == 4] = 255

    mask[:,:,0][mask[:,:,0] == 5] = 255
    mask[:,:,1][mask[:,:,1] == 5] = 0
    mask[:,:,2][mask[:,:,2] == 5] = 255
    
    mask[:,:,0][mask[:,:,0] == 6] = 0
    mask[:,:,1][mask[:,:,1] == 6] = 255
    mask[:,:,2][mask[:,:,2] == 6] = 255

    
    return mask
    



left = int((2560 - 800)/2)
top = int((1440 - 600)/2)

cv2.namedWindow("win1");
cv2.moveWindow("win1", left+795,top-40);

last_time = time.time()
while True:
    screen =  np.array(ImageGrab.grab(bbox=(left, top,left+800,top+600)))
    print('Frame took {} seconds'.format(time.time()-last_time))
    last_time = time.time()
    
    pre_image = preprocess.preprocess_segment_eval(screen)

    pred = model(pre_image.unsqueeze(0).cuda())

    pred = pred.argmax(dim=1)
    
    
    post_pred = F.resize(pred, (280, 800))
    
    
    mask = cv2.cvtColor(post_pred.cpu().to(dtype=torch.uint8).numpy()[0], cv2.COLOR_GRAY2RGB)
    
    
    np.unique(mask.reshape(-1, 3), axis = 0)

    mask = map_colors(mask)

    cv2.imshow('win1',cv2.cvtColor(mask, cv2.COLOR_BGR2RGB))

    if cv2.waitKey(25) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        break