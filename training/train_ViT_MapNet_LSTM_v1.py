from libraries import *
import baseFunctions as bf
from models.ViT_MapNet_LSTM_v1 import *


def create_dataframe(data, group_n=1):
    
  
    idx_order = np.array(list(SubsetRandomSampler(list(BatchSampler(SequentialSampler(data.index), batch_size=group_n, drop_last=True)))), dtype=np.int64)
    
    data_train = data.iloc[idx_order.flatten()]
    
    return data_train



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
    CKP_DIR = "./Data/Models/ViT_MapNet_LSTM_v1/checkpoint/"
    SCORE_DIR = "./Data/Models/ViT_MapNet_LSTM_v1/scores/"
    SCORE_FILE = 'history_score.pkl'
    
    
    if not os.path.exists(CKP_DIR):
        os.makedirs(CKP_DIR)
    
    if not os.path.exists(SCORE_DIR):
        os.makedirs(SCORE_DIR)
        
    
    #data = pd.read_csv(DATA_ROOT_DIR + "data_tot.csv", index_col=0)

    #create_dataframe(data, 10).to_csv('./Data/gta_data/data_tot_seq.csv')
    
    
    
    train_dataset = bf.GTADataset("data_tot_seq.csv", DATA_ROOT_DIR, augment=True, mmap=True)
    
    test_dataset = bf.GTADataset("data_TEST.csv", DATA_ROOT_DIR, mmap=True)
    #test_dataset = bf.GTADataset("temp.csv", DATA_ROOT_DIR, mmap=True)
    
    train_dl = DataLoader(train_dataset, 
                            batch_size=10*10,
                            drop_last = True,
                            num_workers=10)
    
    test_dl = DataLoader(test_dataset, 
                            batch_size=10*10,
                            drop_last = True,
                            num_workers=0)
    


    model = ViT_MapNet_LSTM_v1(num_sequences=10, num_timestep=10).to(device) #qui inserire modello da trainare
    #model.load_state_dict(torch.load(CKP_DIR+ "00100.pth"))
    
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
    sa_pred, sa_gt = trainer.test_model(test_dl)
    
    
    
    figure, ax = plt.subplots(2,1, figsize=(20,10))
    
    ax[0].plot(sa_pred[1:])
    ax[0].plot(sa_gt[1:], alpha=0.5)
    
    
    hs = bf.read_object(SCORE_DIR+"00050_history_score.pkl")

    ax[1].plot(hs["MAE_sa_train"])
    

#== lasts epoch results (train) ==
#

#== Best Result ==
#
