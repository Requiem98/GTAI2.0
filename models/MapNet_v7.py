from libraries import *
import baseFunctions as bf

    
class MapNet_v7(nn.Module):

    def __init__(self, device):

        super().__init__()

        self.device = device
        
        
        #Image
        self.conv1 = nn.Conv2d(3, 24, 5, 2, 0)
        self.batchNorm1 = nn.BatchNorm2d(24)
        self.convlRelu1 = nn.ReLU()
        
        
        self.conv2 = nn.Conv2d(24, 36, 5, 2, 0)
        self.batchNorm2 = nn.BatchNorm2d(36)
        self.convlRelu2 = nn.ReLU()
        
        
        self.conv3 = nn.Conv2d(36, 48, 5, 2, 0)
        self.batchNorm3 = nn.BatchNorm2d(48)
        self.convlRelu3 = nn.ReLU()
        
        
        
        self.conv4 = nn.Conv2d(48, 64, 3, 1, 0)
        self.batchNorm4 = nn.BatchNorm2d(64)
        self.convlRelu4 = nn.ReLU()
        
        
        self.conv5 = nn.Conv2d(64, 64, 3, 1, 0)
        self.batchNorm5 = nn.BatchNorm2d(64)
        self.convlRelu5 = nn.ReLU()
        

        
        
        
        
        
        #MiniMap
        self.conv1_map = nn.Conv2d(3, 24, 3, 2, 0)
        self.batchNorm1_map = nn.BatchNorm2d(24)
        self.convlRelu1_map = nn.ReLU()
        
        
        self.conv2_map = nn.Conv2d(24, 36, 3, 2, 0)
        self.batchNorm2_map = nn.BatchNorm2d(36)
        self.convlRelu2_map = nn.ReLU()
        
        
        
        self.conv3_map = nn.Conv2d(36, 48, 3, 1, 0)
        self.batchNorm3_map = nn.BatchNorm2d(48)
        self.convlRelu3_map = nn.ReLU()
        
        
        self.conv4_map = nn.Conv2d(48, 64, 3, 1, 0)
        self.batchNorm4_map = nn.BatchNorm2d(64)
        self.convlRelu4_map = nn.ReLU()
        
        
        self.conv5_map = nn.Conv2d(64, 64, 3, 1, 0)
        self.batchNorm5_map = nn.BatchNorm2d(64)
        self.convlRelu5_map = nn.ReLU()
        
        

        
        self.flatten = nn.Flatten()
        
        self.linear_speed = nn.Linear(3, 2048)
        self.relu_speed = nn.ReLU()
        
        self.linear1 = nn.Linear(42496, 2048)
        self.batchNorm_linear1 = nn.BatchNorm1d(2048)
        self.relu1 = nn.ReLU()
        
        self.linear2 = nn.Linear(4096, 1024)
        self.batchNorm_linear2 = nn.BatchNorm1d(1024)
        self.relu2 = nn.ReLU()
        
        
        self.drop_1 = nn.Dropout(p=0.1)
        
        self.linear3 = nn.Linear(1024, 512)
        self.batchNorm_linear3 = nn.BatchNorm1d(512)
        self.relu3 = nn.ReLU()
        
        
        self.drop_2 = nn.Dropout(p=0.1)
        
        self.linear4 = nn.Linear(512, 256)
        self.batchNorm_linear4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU()
        
        
        self.drop_3 = nn.Dropout(p=0.1)
        
        self.linear5 = nn.Linear(256, 128)
        self.batchNorm_linear5 = nn.BatchNorm1d(128)
        self.relu5 = nn.ReLU()
        
        
        self.drop_4 = nn.Dropout(p=0.1)
        
        self.linear6 = nn.Linear(128, 64)
        self.batchNorm_linear6 = nn.BatchNorm1d(64)
        self.relu6 = nn.ReLU()
        
        
        self.drop_5 = nn.Dropout(p=0.1)
        
        self.linear7 = nn.Linear(64, 32)
        self.batchNorm_linear7 = nn.BatchNorm1d(32)
        self.relu7 = nn.ReLU()
        
        self.drop_6 = nn.Dropout(p=0.1)
        

        self.linear8 = nn.Linear(32, 2) #steering angle, acceleration
        
        self.linear9 = nn.Linear(32, 1) #brake classes



    def forward(self, x_img, x_mmap, x_speed):

        x_img = self.convlRelu1(self.batchNorm1(self.conv1(x_img)))
        x_img = self.convlRelu2(self.batchNorm2(self.conv2(x_img)))
        x_img = self.convlRelu3(self.batchNorm3(self.conv3(x_img)))
        
        x_img = self.convlRelu4(self.batchNorm4(self.conv4(x_img)))
        x_img = self.convlRelu5(self.batchNorm5(self.conv5(x_img)))

        
        x_img = self.flatten(x_img)
        
        
        x_mmap = self.convlRelu1_map(self.batchNorm1_map(self.conv1_map(x_mmap)))
        x_mmap = self.convlRelu2_map(self.batchNorm2_map(self.conv2_map(x_mmap)))
        x_mmap = self.convlRelu3_map(self.batchNorm3_map(self.conv3_map(x_mmap)))

        x_mmap = self.convlRelu4_map(self.batchNorm4_map(self.conv4_map(x_mmap)))
        x_mmap = self.convlRelu5_map(self.batchNorm5_map(self.conv5_map(x_mmap)))

        x_mmap = self.flatten(x_mmap)
        
        x_speed = x_speed.unsqueeze(1)
        x_speed = torch.cat([x_speed,x_speed**2, x_speed**3], 1)
        
        x_speed = self.relu_speed(self.linear_speed(x_speed))
        
        x = torch.cat([x_img,x_mmap], 1)
        
        x = self.relu1(self.batchNorm_linear1(self.linear1(x)))
        
        x = torch.cat([x,x_speed], 1)
        
        x = self.drop_1(self.relu2(self.batchNorm_linear2(self.linear2(x))))
        x = self.drop_2(self.relu3(self.batchNorm_linear3(self.linear3(x))))
        x = self.drop_3(self.relu4(self.batchNorm_linear4(self.linear4(x))))
        x = self.drop_4(self.relu5(self.batchNorm_linear5(self.linear5(x))))
        x = self.drop_5(self.relu6(self.batchNorm_linear6(self.linear6(x))))
        x = self.drop_6(self.relu7(self.batchNorm_linear7(self.linear7(x))))
        
        x_st_acc = self.linear8(x)
        x_brk = self.linear9(x)

        return x_st_acc, x_brk
     
    

    
    






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
            mae_acc=0
            recall_brk=0

            for id_b, batch in tqdm(enumerate(self.data), total=steps_per_epoch):
                
                optim.zero_grad()

                
                pred, pred_brk = self.model(batch["img"].to(self.model.device), batch["mmap"].to(self.model.device), batch["speed"].to(self.model.device))                  

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)
                gt_acceleration = batch["statistics"][:,1].to(self.model.device)
                gt_brake = batch["brake"].to(self.model.device)

                loss1 = self.mse_loss(pred[:,0].reshape(-1), gt_steeringAngle)
                
                loss2 = self.mse_loss(pred[:,1].reshape(-1), gt_acceleration)
                
                loss3 = self.ce_loss(pred_brk, gt_brake.unsqueeze(1).to(dtype=torch.float32))
                
                loss = loss1 + loss2**2 + loss3**3
                
                with torch.no_grad():
                    train_tot_loss += loss
                    mae_sa += self.mae(pred[:,0].reshape(-1), gt_steeringAngle)
                    mae_acc += self.mae(pred[:,1].reshape(-1), gt_acceleration)
                    recall_brk += self.recall(pred_brk, gt_brake.unsqueeze(1))

                loss.backward()
                optim.step()

                
                
                if(steps_per_epoch == id_b+1):
                    break
            
                
            
            scheduler1.step(train_tot_loss)
            scheduler2.step(mae_sa+mae_acc+recall_brk)

            if (epoch+1) % log_step == 0:
                print('Total Train Loss: %7.10f --- MAE SA: %7.6f --- MAE Acc: %7.6f --- Recall brk: %7.6f' % (train_tot_loss/steps_per_epoch, mae_sa/steps_per_epoch, mae_acc/steps_per_epoch, recall_brk/steps_per_epoch))


            history_score['loss_tot_train'].append((train_tot_loss/steps_per_epoch).item())
            history_score['MAE_sa_train'].append((mae_sa/steps_per_epoch).item())
            history_score['MAE_acc_train'].append((mae_acc/steps_per_epoch).item())
            history_score['Recall_brake_train'].append((recall_brk/steps_per_epoch).item())


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
        recall_brk=0
        
        sa_preds = np.array([0])
        acc_preds = np.array([0])
        brk_preds = np.array([0])
        
        sa_gt = np.array([0])
        acc_gt = np.array([0])
        brk_gt = np.array([0])
        
        for id_b, batch in tqdm(enumerate(test_data), total=len(test_data)):
            

            with torch.no_grad():
                
                pred, pred_brk = self.model(batch["img"].to(self.model.device), batch["mmap"].to(self.model.device), batch["speed"].to(self.model.device))                     

                gt_steeringAngle = batch["statistics"][:,0].to(self.model.device)
                gt_acceleration = batch["statistics"][:,1].to(self.model.device)
                gt_brake = batch["brake"].to(self.model.device)

                loss1 = self.mse_loss(pred[:,0].reshape(-1), gt_steeringAngle)
                
                loss2 = self.mse_loss(pred[:,1].reshape(-1), gt_acceleration)
                
                loss3 = self.ce_loss(pred_brk, gt_brake.unsqueeze(1).to(dtype=torch.float32))
                
                loss = loss1 + loss2**2 + loss3**3
                
                
                
                test_tot_loss += loss
                mae_sa += self.mae(pred[:,0].reshape(-1), gt_steeringAngle)
                mae_acc += self.mae(pred[:,1].reshape(-1), gt_acceleration)
                recall_brk += self.recall(pred_brk, gt_brake.unsqueeze(1))
                
                
                pred_brk = (torch.sigmoid(pred_brk.flatten()) > 0.5).to(dtype=torch.int32).cpu().numpy()
                
                sa_preds = np.concatenate([sa_preds, pred[:,0].cpu().numpy().flatten()])
                acc_preds = np.concatenate([acc_preds, pred[:,1].cpu().numpy().flatten()])
                brk_preds = np.concatenate([brk_preds, pred_brk])
                
                sa_gt = np.concatenate([sa_gt, gt_steeringAngle.cpu().numpy().flatten()])
                acc_gt = np.concatenate([acc_gt, gt_acceleration.cpu().numpy().flatten()])
                brk_gt = np.concatenate([brk_gt, gt_brake.cpu().numpy().flatten()])
                
                
        print('Total Test Loss: %7.10f --- MAE SA: %7.6f --- MAE Acc: %7.6f --- Recall brk: %7.6f' % (test_tot_loss/len(test_data), mae_sa/len(test_data), mae_acc/len(test_data), recall_brk/len(test_data)))
                
        return sa_preds, acc_preds, brk_preds, sa_gt, acc_gt, brk_gt
    
    
    def mse_loss(self, pred, target):
        return torch.nn.functional.mse_loss(pred, target)
    
    def mae(self, pred, target):
        return torch.nn.functional.l1_loss(pred, target)
    
    def ce_loss(self, pred, target):
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, target)
    
    def recall(self, pred, target):
        return torchmetrics.functional.classification.binary_recall(pred, target)
