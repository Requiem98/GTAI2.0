from libraries import *
import baseFunctions as bf
from models.modules.general import *
    
class MapNet_v3(nn.Module):

    def __init__(self):

        super().__init__()

        
        
        
        self.conv1 = CONV_BLOCK(3, 24, 5, 2, "valid", 1)   #Bx24x110x110
        self.conv2 = CONV_BLOCK(24, 36, 5, 2, "valid", 1)  #Bx24x53x53
        self.conv3 = CONV_BLOCK(36, 48, 5, 2, "valid", 1)  #Bx24x25x25
        
        self.conv4 = CONV_BLOCK(48, 64, 3, 2, "valid", 1)  #Bx24x23x23
        self.conv5 = CONV_BLOCK(64, 64, 3, 2, "valid", 1)  #Bx24x21x21
        
        self.flatten = nn.Flatten()
       
        #MiniMap
        self.conv_map = MMAP_CONV() #8192
        
        
        self.mlp = MLP(in_dim = 36416, hidden_dims = [4096, 1024, 32])

        self.head = nn.Linear(32, 1) #steering angle




    def forward(self, x_img, x_mmap):

        x_img = self.conv1(x_img)
        x_img = self.conv2(x_img)
        x_img = self.conv3(x_img)
        x_img = self.conv4(x_img)
        x_img = self.conv5(x_img)
        
        x_img = self.flatten(x_img)
        
        x_mmap = self.conv_map(x_mmap)
        
        x = torch.cat([x_img,x_mmap], 1)
        
        x = self.mlp(x)

        return self.head(x)
     

    
    
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
            
       
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=2, verbose=True)  # goal: minimize loss
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=2, verbose=True)  # goal: minimize mae
            
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

                
                pred = self.model(batch["img"].to(self.device), batch["mmap"].to(self.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.device)

                loss = self.mse_loss(pred.reshape(-1), gt_steeringAngle)
                
                with torch.no_grad():
                    train_tot_loss += loss
                    mae_sa += self.mae(pred.reshape(-1), gt_steeringAngle)

                loss.backward()
                optim.step()

                
                
                if(steps_per_epoch == id_b+1):
                    break
            
                
            
            scheduler1.step(train_tot_loss)
            scheduler2.step(mae_sa)

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

                gt_steeringAngle = batch["statistics"][:,0].to(self.device)

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




