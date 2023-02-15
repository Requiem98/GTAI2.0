from libraries import *
import baseFunctions as bf
from models.MapNet_v3 import *





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
    CKP_DIR = "./Data/models/MapNet_v3/checkpoint/"
    SCORE_DIR = "./Data/models/MapNet_v3/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)
    
    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)
    
    train_dataset = bf.GTADataset("data_tot.csv", DATA_ROOT_DIR, augment=True, mmap=True)
    
    test_dataset = bf.GTADataset("data_TEST.csv", DATA_ROOT_DIR, mmap=True)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=256, 
                            shuffle=True,
                            num_workers=10)
    
    test_dl = DataLoader(test_dataset, 
                            batch_size=256, 
                            num_workers=2)
    

    model = MapNet_v3().to(device) #qui inserire modello da trainare
    #model.load_state_dict(torch.load(CKP_DIR+ "00035.pth"))
    
    
    trainer = Trainer(device, model, 
                      ckp_dir = CKP_DIR, 
                      score_dir = SCORE_DIR, 
                      score_file = SCORE_FILE)
    
    """
    trainer.train_model(train_dl,
                        max_epoch=40, 
                        steps_per_epoch=0,
                        lr=0.01,
                        weight_decay=0,
                        log_step=1, 
                        ckp_save_step = 5,
                        ckp_epoch=10)
    """
    
    print('Starting test...')
    sa_pred, sa_gt = trainer.test_model(test_dl)
    
    
    
    figure, ax = plt.subplots(2,1, figsize=(20,10))
    
    ax[0].plot(sa_pred[1:])
    ax[0].plot(sa_gt[1:], alpha=0.5)
    
    
    
    hs = bf.read_object(SCORE_DIR+"00050_history_score.pkl")

    ax[1].plot(hs["MAE_sa_train"])
    

#== lasts epoch results (train) ==
#Total Train Loss: 0.0010056252 --- MAE SA: 0.023171 #epoch 10
#Total Train Loss: 0.0001449459 --- MAE SA: 0.008264 #epoch 50

#== Best Result ==
#Total Test Loss: 0.0029799272 --- MAE SA: 0.035191 #epoch 10
#Total Test Loss: 0.0025588726 --- MAE SA: 0.032796 #epoch 20
#Total Test Loss: 0.0031932767 --- MAE SA: 0.037306 #epoch 50

    

