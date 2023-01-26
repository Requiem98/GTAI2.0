from libraries import *
import baseFunctions as bf
from models.MapNet_v8 import *





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
    CKP_DIR = "./Data/models/MapNet_v8/checkpoint/"
    SCORE_DIR = "./Data/models/MapNet_v8/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)
    
    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)
    
    train_dataset = bf.GTADataset("data_tot.csv", DATA_ROOT_DIR, bf.preprocess, mmap=True)
    
    test_dataset = bf.GTADataset("data_TEST.csv", DATA_ROOT_DIR, bf.test_preprocess, mmap=True)
    #test_dataset = bf.GTADataset("temp.csv", DATA_ROOT_DIR, bf.test_preprocess, mmap=True)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=512, 
                            shuffle=True,
                            num_workers=10)
    
    test_dl = DataLoader(test_dataset, 
                            batch_size=512, 
                            num_workers=0)
    


    model = MapNet_v8(device = device).to(device) #qui inserire modello da trainare
    model.load_state_dict(torch.load(CKP_DIR+ "00025.pth"))
    
    trainer = Trainer(model, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    """
    trainer.train_model(train_dl,
                        max_epoch=40, 
                        steps_per_epoch=0,
                        lr=0.001,
                        weight_decay=0,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=10)
    """
    
    print('Starting test...')
    sa_pred, acc_pred, brk_pred, sa_gt, acc_gt, brk_gt = trainer.test_model(test_dl)
    
    
    
    figure, ax = plt.subplots(2,3, figsize=(20,10))
    
    ax[0, 0].plot(sa_pred[1:])
    ax[0, 0].plot(sa_gt[1:], alpha=0.5)
    
    
    ax[0, 1].plot(acc_pred[1:])
    ax[0, 1].plot(acc_gt[1:], alpha=0.5)
    
    ax[0, 2].plot(brk_pred[1:])
    ax[0, 2].plot(brk_gt[1:], alpha=0.5)
    
    
    hs = bf.read_object(SCORE_DIR+"00025_history_score.pkl")

    ax[1, 0].plot(hs["MAE_sa_train"])
    ax[1, 1].plot(hs["MAE_acc_train"])
    ax[1, 2].plot(hs["Recall_brake_train"])
    

#== lasts epoch results (train) ==
#Total Train Loss: 0.0031270436 --- MAE SA: 0.030678 --- MAE Acc: 0.127504 --- Recall brk: 0.875765 # epoch 50

#== Best Result ==
#Total Test Loss: 0.0528299138 --- MAE SA: 0.048327 --- MAE Acc: 0.161320 --- Recall brk: 0.335000 # epoch 50
