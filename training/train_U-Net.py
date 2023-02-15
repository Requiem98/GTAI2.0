from libraries import *
import baseFunctions as bf
from torchvision.transforms import functional as f
from Models.Unet import *


if __name__ == '__main__':
    
    torch.multiprocessing.set_start_method('spawn')
    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))
        bf.get_memory()
        
    
    #Path per i salvataggi dei checkpoints
    #SINTASSI: ./Data/models/NOME_MODELLO/etc...
    CKP_DIR = "./Data/Models/Unet/checkpoint/"
    SCORE_DIR = "./Data/Models/Unet/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)
    
    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)
    
    
    train_dataset = bf.GTA_segment_Dataset(csv_file = "segment_data_train.csv", root_dir="./Data/segmentation_dataset/")
    test_dataset  = bf.GTA_segment_Dataset(csv_file = "segment_data_test.csv", root_dir="./Data/segmentation_dataset/")
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=16,
                            shuffle=True,
                            num_workers=10)
    
    
    test_dl = DataLoader(test_dataset, 
                            batch_size=16, 
                            num_workers=2)

    
    model = UNet(n_channels=3, n_classes=7).to(device)
    model.load_state_dict(torch.load(CKP_DIR+ "00050.pth"))
    
    
    trainer = Trainer(device, model, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    
    trainer.train_model(train_dl,
                        test_dl,
                        max_epoch=100, 
                        steps_per_epoch=0,
                        lr=0.01,
                        weight_decay=1e-10,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=0)
    
    
    dice_score = trainer.test_model(test_dl)
    print(f"\n\n\nTEST DICE SCORE: {dice_score}\n\n")
    
    
#== lasts epoch results (train) ==
#--- Total Train Loss:  0.0500  --- Dice score:  0.8063 #epoch 100

#== Best Result ==
#TEST DICE SCORE: 0.7501963973045349 # epoch 100   
    
preprocess = bf.PREPROCESS()
model.eval()

    
data = pd.read_csv(DATA_ROOT_DIR + "data_tot.csv", index_col = 0)

image = io.imread(data["path"][500])

plt.imshow(image)
    
orig_image = image[200:480]

plt.imshow(orig_image)

orig_image.shape    

pre_image = preprocess.preprocess_segment_eval(image)


#image = test_dataset.__getitem__(400)["img"]
#mask = test_dataset.__getitem__(3)["mask"]

plt.imshow(pre_image.numpy().transpose(1,2,0))



pred = model(pre_image.unsqueeze(0).cuda())

pred = pred.argmax(dim=1)

pred.flatten().shape

figure, ax = plt.subplots(2,1, figsize=(20,10))
ax[0].imshow(pre_image.numpy().transpose(1,2,0))
ax[1].imshow(pred.cpu()[0])



post_image = F.resize(pre_image, (280, 800))
post_pred = F.resize(pred, (280, 800))

figure, ax = plt.subplots(2,1, figsize=(20,10))
ax[0].imshow(post_image.numpy().transpose(1,2,0))
ax[1].imshow(post_pred.cpu()[0])



"""
figure, ax = plt.subplots(3,1)
ax[0].imshow(image.numpy().transpose(1,2,0))
ax[1].imshow(mask)
ax[2].imshow(pred.cpu()[0])

figure, ax = plt.subplots(2,1)
ax[0].imshow(mask[:40, 180:200])
ax[1].imshow(pred.cpu()[0][:40, 180:200])


hs = bf.read_object(SCORE_DIR+"00100_history_score.pkl")

plt.plot(hs["loss_train"])
plt.plot(torch.tensor(hs["dice_score"]).cpu())
"""
    
     
    
    
    