from libraries import *
import baseFunctions as bf

    
class MapNet_v4(nn.Module):

    def __init__(self, device):

        super().__init__()

        self.device = device
        
        
        #Image
        self.conv1 = nn.Conv2d(3, 24, 5, 2, 0)
        self.convlRelu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(24)
        
        self.conv2 = nn.Conv2d(24, 36, 5, 2, 0)
        self.convlRelu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(36)
        
        self.conv3 = nn.Conv2d(36, 48, 5, 2, 0)
        self.convlRelu3 = nn.ReLU()
        self.batchNorm3 = nn.BatchNorm2d(48)
        
        
        self.conv4 = nn.Conv2d(48, 64, 3, 1, 0)
        self.convlRelu4 = nn.ReLU()
        self.batchNorm4 = nn.BatchNorm2d(64)
        
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        self.convlRelu5 = nn.ReLU()
        self.batchNorm5 = nn.BatchNorm2d(64)

        
        
        
        
        
        #MiniMap
        self.conv1_map = nn.Conv2d(3, 24, 3, 2, 0)
        self.convlRelu1_map = nn.ReLU()
        self.batchNorm1_map = nn.BatchNorm2d(24)
        
        self.conv2_map = nn.Conv2d(24, 36, 3, 2, 0)
        self.convlRelu2_map = nn.ReLU()
        self.batchNorm2_map = nn.BatchNorm2d(36)
        
        
        self.conv3_map = nn.Conv2d(36, 48, 3, 1, 0)
        self.convlRelu3_map = nn.ReLU()
        self.batchNorm3_map = nn.BatchNorm2d(48)
        
        self.conv4_map = nn.Conv2d(48, 64, 3, 1, 0)
        self.convlRelu4_map = nn.ReLU()
        self.batchNorm4_map = nn.BatchNorm2d(64)
        
        self.conv5_map = nn.Conv2d(64, 64, 3, 1, 0)
        self.convlRelu5_map = nn.ReLU()
        self.batchNorm5_map = nn.BatchNorm2d(64)
        

        
        self.flatten = nn.Flatten()
        
        
        self.linear1 = nn.Linear(42497, 2048)
        self.relu1 = nn.ReLU()
        self.batchNorm_linear1 = nn.BatchNorm1d(2048)
        
        self.linear2 = nn.Linear(2048, 1024)
        self.relu2 = nn.ReLU()
        self.batchNorm_linear2 = nn.BatchNorm1d(1024)
        
        self.linear3 = nn.Linear(1024, 512)
        self.relu3 = nn.ReLU()
        self.batchNorm_linear3 = nn.BatchNorm1d(512)
        
        self.linear4 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.batchNorm_linear4 = nn.BatchNorm1d(256)
        
        self.linear5 = nn.Linear(256, 128)
        self.relu5 = nn.ReLU()
        self.batchNorm_linear5 = nn.BatchNorm1d(128)
        
        self.linear6 = nn.Linear(128, 64)
        self.relu6 = nn.ReLU()
        self.batchNorm_linear6 = nn.BatchNorm1d(64)
        
        self.linear7 = nn.Linear(64, 32)
        self.relu7 = nn.ReLU()
        self.batchNorm_linear7 = nn.BatchNorm1d(32)

        self.linear8 = nn.Linear(32, 3) #steering angle, acceleration, brake




    def forward(self, x_img, x_mmap, x_speed):

        x_img = self.batchNorm1(self.convlRelu1(self.conv1(x_img)))
        x_img = self.batchNorm2(self.convlRelu2(self.conv2(x_img)))
        x_img = self.batchNorm3(self.convlRelu3(self.conv3(x_img)))

        x_img = self.batchNorm4(self.convlRelu4(self.conv4(x_img)))
        x_img = self.batchNorm5(self.convlRelu5(self.conv5(x_img)))
        
        x_img = self.flatten(x_img)
        
        
        x_mmap = self.batchNorm1_map(self.convlRelu1_map(self.conv1_map(x_mmap)))
        x_mmap = self.batchNorm2_map(self.convlRelu2_map(self.conv2_map(x_mmap)))
        x_mmap = self.batchNorm3_map(self.convlRelu3_map(self.conv3_map(x_mmap)))

        x_mmap = self.batchNorm4_map(self.convlRelu4_map(self.conv4_map(x_mmap)))
        x_mmap = self.batchNorm5_map(self.convlRelu5_map(self.conv5_map(x_mmap)))

        x_mmap = self.flatten(x_mmap)
        
        
        x = torch.cat([x_img,x_mmap, x_speed.unsqueeze(1)], 1)
        
        x = self.batchNorm_linear1(self.relu1(self.linear1(x)))
        x = self.batchNorm_linear2(self.relu2(self.linear2(x)))
        x = self.batchNorm_linear3(self.relu3(self.linear3(x)))
        x = self.batchNorm_linear4(self.relu4(self.linear4(x)))
        x = self.batchNorm_linear5(self.relu5(self.linear5(x)))
        x = self.batchNorm_linear6(self.relu6(self.linear6(x)))
        x = self.batchNorm_linear7(self.relu7(self.linear7(x)))
        
        x = self.linear8(x)

        return x
     
    
    
    def loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
    
    def MeanAbsoluteError(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)
    
    






class Trainer():
    
    def __init__(self, model, ckp_dir = "", score_dir = "", score_file = "score.pkl"):
        self.model = model
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
            
       
        scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=2, verbose=True)  # goal: maximize loss
        scheduler2 = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'min', patience=2, verbose=True)  # goal: maximize mae
            
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
            mae_acc=0
            mae_brk=0

            for id_b, batch in tqdm(enumerate(self.data), total=steps_per_epoch):
                
                optim.zero_grad()

                
                pred = self.model(batch["img"].to(self.model.device), batch["mmap"].to(self.model.device), batch["speed"].to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)
                gt_acceleration = batch["statistics"][:,1].to(self.model.device)
                gt_brake = batch["statistics"][:,2].to(self.model.device)

                loss1 = self.model.loss(pred[:,0].reshape(-1), gt_steeringAngle)
                
                loss2 = self.model.loss(pred[:,1].reshape(-1), gt_acceleration)
                
                loss3 = self.model.loss(pred[:,2].reshape(-1), gt_brake)
                
                loss = loss1 + loss2+ loss3
                
                with torch.no_grad():
                    train_tot_loss += loss
                    mae_sa += self.model.MeanAbsoluteError(pred[:,0].reshape(-1), gt_steeringAngle)
                    mae_acc += self.model.MeanAbsoluteError(pred[:,1].reshape(-1), gt_acceleration)
                    mae_brk += self.model.MeanAbsoluteError(pred[:,2].reshape(-1), gt_brake)

                loss.backward()
                optim.step()

                
                
                if(steps_per_epoch == id_b):
                    break
            
                
            
            scheduler1.step(train_tot_loss)
            scheduler2.step(mae_sa+mae_acc+mae_brk)

            if (epoch+1) % log_step == 0:
                print('Total Train Loss: %7.4f --- MAE SA: %7.4f --- MAE Acc: %7.4f --- MAE brk: %7.4f' % (train_tot_loss/steps_per_epoch, mae_sa/steps_per_epoch, mae_acc/steps_per_epoch, mae_brk/steps_per_epoch))


            history_score['loss_tot_train'].append((train_tot_loss/steps_per_epoch).item())
            history_score['MAE_sa_train'].append((mae_sa/steps_per_epoch).item())
            history_score['MAE_acc_train'].append((mae_acc/steps_per_epoch).item())
            history_score['MAE_brake_train'].append((mae_brk/steps_per_epoch).item())


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
        mae_acc=0
        mae_brk=0
        
        sa_preds = np.array([0])
        acc_preds = np.array([0])
        brk_preds = np.array([0])
        
        sa_gt = np.array([0])
        acc_gt = np.array([0])
        brk_gt = np.array([0])
        
        for id_b, batch in tqdm(enumerate(test_data), total=len(test_data)):
            

            with torch.no_grad():
                
                pred = self.model(batch["img"].to(self.model.device), batch["mmap"].to(self.model.device), batch["speed"].to(self.model.device))                     

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)
                gt_acceleration = batch["statistics"][:,1].to(self.model.device)
                gt_brake = batch["statistics"][:,2].to(self.model.device)

                loss1 = self.model.loss(pred[:,0].reshape(-1), gt_steeringAngle)
                
                loss2 = self.model.loss(pred[:,1].reshape(-1), gt_acceleration)
                
                loss3 = self.model.loss(pred[:,2].reshape(-1), gt_brake)
                
                loss = loss1 + loss2+ loss3
                
                
                
                test_tot_loss += loss
                mae_sa += self.model.MeanAbsoluteError(pred[:,0].reshape(-1), gt_steeringAngle)
                mae_acc += self.model.MeanAbsoluteError(pred[:,1].reshape(-1), gt_acceleration)
                mae_brk += self.model.MeanAbsoluteError(pred[:,2].reshape(-1), gt_brake)
                
                
                
                sa_preds = np.concatenate([sa_preds, pred[:,0].cpu().numpy().flatten()])
                acc_preds = np.concatenate([acc_preds, pred[:,1].cpu().numpy().flatten()])
                brk_preds = np.concatenate([brk_preds, pred[:,2].cpu().numpy().flatten()])
                
                sa_gt = np.concatenate([sa_gt, gt_steeringAngle.cpu().numpy().flatten()])
                acc_gt = np.concatenate([acc_gt, gt_acceleration.cpu().numpy().flatten()])
                brk_gt = np.concatenate([brk_gt, gt_brake.cpu().numpy().flatten()])
                
                
        print('Total Test Loss: %7.4f --- MAE SA: %7.4f --- MAE Acc: %7.4f --- MAE brk: %7.4f' % (test_tot_loss/len(test_data), mae_sa/len(test_data), mae_acc/len(test_data), mae_brk/len(test_data)))
                
        return sa_preds, acc_preds, brk_preds, sa_gt, acc_gt, brk_gt
