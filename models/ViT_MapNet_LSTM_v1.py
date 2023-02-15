from libraries import *
import baseFunctions as bf
from models.modules.ViT import ViT
from models.modules.general import *



        

class ViT_MapNet_LSTM_v1(nn.Module):

    def __init__(self, num_sequences, num_timestep):

        super().__init__()

        self.num_sequences = num_sequences
        self.num_timestep = num_timestep
        
        
        #===========================
        #========== Image ==========
        #===========================
        

        self.vit = ViT(image_size=224, patch_size=16, d_model=512, nhead=8, num_layers=6) #Bx197x512

        self.linear = nn.Linear(512, 125) #Bx197x125
        self.relu = nn.ReLU()
        
        self.flat = nn.Flatten() #Bx24625 (~ 75% of 8192+24625)
        

        #===========================
        #=========  MMAP ===========
        #===========================
        
        self.conv_map = MMAP_CONV() #Bx8192 (~ 25% of 8192+24625)
        
        
        #===========================
        #========= LSTM ============
        #===========================
        
        self.lstm = nn.LSTM(input_size = 32817, hidden_size = 256, num_layers = 2, dropout = 0.1)
        
        self.mlp = MLP(256, [128, 64, 32])

        self.head = nn.Linear(32, 1) #steering angle



    def forward(self, x_img, x_mmap, x_speed):
        
        x_img = self.flat(self.relu(self.linear(self.vit(x_img, x_speed)))) #Bx24625
        
        x_mmap = self.conv_map(x_mmap) #Bx8192
        
        x = self._process_features(x_img, x_mmap)
        
        x, _ = self.lstm(x) #(S,B,D)
        
        x = x[-1]     #(B, D)
        
        x = self.mlp(x)
           

        return self.head(x)
    
    def _process_features(self, x_img, x_mmap):
        
        x = torch.cat([x_img, x_mmap], axis = 1) #Bx(8192+24625) = Bx32817
    
        x = x.reshape(self.num_timestep, self.num_sequences, x.shape[1]) #(S, B, D)
        
        return x
        
        
    
    






class Trainer():
    
    def __init__(self, device, model, ckp_dir = "", score_dir = "", score_file = "score.pkl"):
        self.model = model
        self.device = device
        self.ckp_dir = ckp_dir
        self.score_dir= score_dir
        self.score_file = score_file
        
    def train_model(self, data, max_epoch=40, steps_per_epoch=0, lr = 1e-3, lr_cap=1e-3, weight_decay = 0, ckp_save_step=20, log_step=5, ckp_epoch=0):

       # Argument for the training
       #max_epoch          # Total number of epoch
       #ckp_save_step      # Frequency for saving the model
       #log_step           # Frequency for printing the loss
       #lr                 # Learning rate
       #weight_decay       # Weight decay
       #ckp_dir            # Directory where to save the checkpoints
       #score_dir          # Directory where to save the scores
       #score_file         # Name of the scores file
       #ckp_epoch          # If the checkpoint file is passed, this indicate the checkpoint training epoch
       #ckp_epoch          # Load weights from indicated epoch if a corresponding checkpoint file is present
       
        self.data = data
       
        if(steps_per_epoch==0):
            steps_per_epoch=len(self.data)
       
        optim = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay = weight_decay)

        if(ckp_epoch != 0):
            self.model.load_state_dict(torch.load(self.ckp_dir + f'{(ckp_epoch):05d}.pth'))
            optim.load_state_dict(torch.load(self.ckp_dir + f'optim_{(ckp_epoch):05d}.pth'))
            history_score = bf.read_object(self.score_dir + f'{(ckp_epoch):05d}_' + self.score_file)
        else:
            history_score = defaultdict(list)
            
       
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=3, verbose=True, threshold = 1e-5, min_lr = 1e-8)
            
        torch.backends.cudnn.benchmark = True
        
        # compute execution time of the cell
        start_time = time.time()




        print("Start Training...\n")


        for epoch in range(max_epoch):

            if (epoch+1) % log_step == 0:
                print("---> Epoch %03i/%03i <--- " % ((epoch+1), max_epoch))

            ###### TRAIN ######
            self.model.train()

            train_tot_loss = 0
            mae_sa=0

            for id_b, batch in tqdm(enumerate(self.data), total=steps_per_epoch):
                
                optim.zero_grad()

                
                pred = self.model(batch["img"].to(self.device), batch["mmap"].to(self.device), batch["speed"].to(self.device).unsqueeze(1))                  

                gt_steeringAngle = batch["statistics"][9::10,0].to(self.device)

                loss = self.mse_loss(pred.reshape(-1), gt_steeringAngle)
                
                
                with torch.no_grad():
                    train_tot_loss += loss
                    mae_sa += self.mae(pred.reshape(-1), gt_steeringAngle)

                loss.backward()
                optim.step()

                
                
                if(steps_per_epoch == id_b+1):
                    break
            
                
            
            scheduler.step(train_tot_loss+mae_sa)

            if (epoch+1) % log_step == 0:
                print('Total Train Loss: %7.10f --- MAE SA: %7.6f' % (train_tot_loss/steps_per_epoch, mae_sa/steps_per_epoch))


            history_score['loss_tot_train'].append((train_tot_loss/steps_per_epoch).item())
            history_score['MAE_sa_train'].append((mae_sa/steps_per_epoch).item())


            # Here we save checkpoints to avoid repeated training
            if ((epoch+1) % (ckp_save_step) == 0):
                print("Saving checkpoint... \n ")
                torch.save(self.model.state_dict(), self.ckp_dir + f'{(epoch+1+ckp_epoch):05d}.pth')
                torch.save(optim.state_dict(), self.ckp_dir + f'optim_{(epoch+1+ckp_epoch):05d}.pth')
                bf.save_object(history_score, self.score_dir + f'{(epoch+1+ckp_epoch):05d}_' + self.score_file)
                


        print("Saving checkpoint... \n ")
        torch.save(self.model.state_dict(), self.ckp_dir + f'{(epoch+1+ckp_epoch):05d}.pth')
        torch.save(optim.state_dict(), self.ckp_dir + f'optim_{(epoch+1+ckp_epoch):05d}.pth')
        bf.save_object(history_score, self.score_dir + f'{(epoch+1+ckp_epoch):05d}_' + self.score_file)

        # print execution time
        print("Total time: %s seconds" % (time.time() - start_time))
        
        
    def test_model(self, test_data):
        
        self.model.eval()
        
        test_tot_loss=0
        mae_sa=0
        
        sa_preds = np.array([0])
        sa_gt = np.array([0])
        
        for id_b, batch in tqdm(enumerate(test_data), total=len(test_data)):
            

            with torch.no_grad():
                
                pred = self.model(batch["img"].to(self.device), batch["mmap"].to(self.device), batch["speed"].to(self.device).unsqueeze(1))                     

                gt_steeringAngle = batch["statistics"][9::10,0].to(self.device)

                loss = self.mse_loss(pred.reshape(-1), gt_steeringAngle)
                
                
                
                test_tot_loss += loss
                mae_sa += self.mae(pred.reshape(-1), gt_steeringAngle)
                
                
                sa_preds = np.concatenate([sa_preds, pred[:,0].cpu().numpy().flatten()])
                
                sa_gt = np.concatenate([sa_gt, gt_steeringAngle.cpu().numpy().flatten()])
                
                
        print('Total Test Loss: %7.10f --- MAE SA: %7.6f' % (test_tot_loss/len(test_data), mae_sa/len(test_data)))
                
        return sa_preds, sa_gt
    
    
    def mse_loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
    
    def mae(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)
    
