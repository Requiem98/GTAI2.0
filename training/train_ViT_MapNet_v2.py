from libraries import *
import baseFunctions as bf
from Models.ViT_MapNet_v2 import *





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
    CKP_DIR = "./Data/Models/ViT_MapNet_v2/checkpoint/"
    SCORE_DIR = "./Data/Models/ViT_MapNet_v2/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)
    
    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)
    
    train_dataset = bf.GTADataset("data_tot.csv", DATA_ROOT_DIR, augment=True, mmap=True)
    
    test_dataset = bf.GTADataset("data_TEST.csv", DATA_ROOT_DIR, mmap=True)
    #test_dataset = bf.GTADataset("temp.csv", DATA_ROOT_DIR, mmap=True)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=20, 
                            shuffle=True,
                            num_workers=10)
    
    test_dl = DataLoader(test_dataset, 
                            batch_size=32, 
                            num_workers=2)
    


    model = ViT_MapNet_v2().to(device) #qui inserire modello da trainare
    #model.load_state_dict(torch.load(CKP_DIR+ "00020.pth"), strict = False)
    
    trainer = Trainer(device, model, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    
    trainer.train_model(train_dl,
                        max_epoch=50, 
                        steps_per_epoch=0,
                        lr=0.01,
                        weight_decay=0,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=0)
    
    
    print('Starting test...')
    sa_pred, sa_gt= trainer.test_model(test_dl)
    
    
    
    figure, ax = plt.subplots(2,1, figsize=(10,5))
    
    ax[0].plot(sa_pred[1:])
    ax[0].plot(sa_gt[1:], alpha=0.5)
    
    
    hs = bf.read_object(SCORE_DIR+"00150_history_score.pkl")

    ax[1].plot(hs["MAE_sa_train"])
    

#== lasts epoch results (train) ==
#Total Train Loss: 0.0146200145 --- MAE SA: 0.034466 --- MAE Acc: 0.043292 --- Recall brk: 0.776287

#== Best Result ==
#Total Test Loss: 1.5395251513 --- MAE SA: 0.041241 --- MAE Acc: 0.161522 --- Recall brk: 0.025774 # epoch 50
#Total Test Loss: 1.2924324274 --- MAE SA: 0.038663 --- MAE Acc: 0.153047 --- Recall brk: 0.025332 # epoch 100
#Total Test Loss: 1.3964891434 --- MAE SA: 0.043396 --- MAE Acc: 0.156967 --- Recall brk: 0.036304 # epoch 150 +data
