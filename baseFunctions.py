from libraries import *
from torch import Tensor

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)
        
def read_object(filename):
    with open(filename, 'rb') as file:
        data = pickle.load(file)
    return data

def get_memory():
    t = torch.cuda.get_device_properties(0).total_memory
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    f = r-a  # free inside reserved

    print("Total memory:", t/1000/1000/1000)
    print("Reserved memory:", r/1000/1000/1000)
    print("Allocated memory:", a/1000/1000/1000)
    print("Free memory:", f/1000/1000/1000)

def free_memory():
    gc.collect()
    torch.cuda.empty_cache()


def cap_hist(data, k, category = "steering_angle"):
    
    new_data = data.copy()
    
    values, bins = np.histogram(new_data[category], bins=80)
    
    for i, count in enumerate(values):
        if(count > k):
            new_data.drop(new_data.loc[(new_data[category] >= bins[i] ) & (new_data[category] <bins[i+1])].sample(n=count-k, random_state=1).index, axis=0, inplace=True)

    
    new_data[category].hist(bins=80)

    return new_data


def reuse_weights(path, model):
    weights = torch.load(path)
    model_dict = model.state_dict()
    
    final_weights = { k:v for k,v in weights.items() if k in model_dict and model_dict[k].shape == v.shape}
    
    model.load_state_dict(final_weights, strict=False)
    return model


def quantized_stat(data, category = "brake", bins=[-0.1, 0.001, 1]):
    
    new_data = data.copy()
    
    new_data[category + "_bin"] = new_data[category]
    
    values, bins = np.histogram(new_data[category], bins=bins)
    
    for i, count in enumerate(values):
        new_data[category + "_bin"].loc[(new_data[category + "_bin"] >= bins[i] ) & (new_data[category + "_bin"] <bins[i+1])] = i


    return new_data



def create_train_test_dataframe(data, group_n=1, test_size=0.2, save_dir = "./Data/", test_file_name = "data_test.csv", train_file_name = "data_train.csv", save=True):
    
    idx_order = np.array(list(SubsetRandomSampler(list(BatchSampler(SequentialSampler(data.index), batch_size=group_n, drop_last=True)))), dtype=np.int64)
    
    idx_tr, idx_test = train_test_split(idx_order, test_size=test_size)
    
    data_train = data.iloc[idx_tr.flatten()]
    
    data_test = data.iloc[idx_test.flatten()]
    
    if(save):
        data_train.to_csv(save_dir + train_file_name)
        data_test.to_csv(save_dir + test_file_name)
    
    return data_train, data_test



#==============================================================================
#================================ Preprocess ==================================
#============================================================================== 


class PREPROCESS:
    def __init__(self):
        
        self.augment = T.Compose([
            T.ColorJitter(brightness=(0.5,2)),
            T.RandomAffine(degrees = 0, translate=(0.1,0.1))
        ])
    
    def preprocess_augment_image(self, image):
        image = image[200:480, :]
        image = F.to_pil_image(image)
        image = F.resize(image, (224,224))
        image = self.augment(image)
        image = F.to_tensor(image)
        return image
    
    def preprocess_image(self, image):
        image = image[200:480, :]
        image = F.to_pil_image(image)
        image = F.resize(image, (224,224))
        image = F.to_tensor(image)
        return image
    
    def preprocess_mmap(self, mmap):
        mmap = mmap[500:580, 50:130]
        mmap = F.to_pil_image(mmap)
        mmap = F.resize(mmap, (40,40))
        mmap = F.to_tensor(mmap)
        return mmap
    
    def preprocess_segment(self, image):
        image = image[400:900]
        image = F.to_pil_image(image)
        image = F.resize(image, (224,224))
        image = F.to_tensor(image)
        return image
    
    def preprocess_segment_eval(self, image):
        image = image[200:480]
        image = F.to_pil_image(image)
        image = F.resize(image, (224,224))
        image = F.to_tensor(image)
        return image
    
    def preprocess_mask(self, mask):
        mask = mask[400:900]
        mask = F.to_pil_image(mask)
        mask = F.resize(mask, (224,224))
        mask = torch.tensor(np.array(mask), dtype=torch.int64)
        #remove not usefull classes
        mask[((mask != 7) & (mask != 24) & (mask != 26) & (mask != 28) & (mask != 32) & (mask != 33))] = 0
        
        mask[(mask == 7)] = 1
        mask[(mask == 24)] = 2
        mask[(mask == 26)] = 3
        mask[(mask == 28)] = 4
        mask[(mask == 32)] = 5
        mask[(mask == 33)] = 6
        
        
        return mask
    
    def preprocess_image_predict(self, image):
        return self.preprocess_image(image).unsqueeze(0)
    
    def preprocess_mmap_predict(self, mmap):
        return self.preprocess_mmap(mmap).unsqueeze(0)

    
#============================================================================
#================================ Dataset ===================================
#============================================================================ 
    
class GTADataset(Dataset):

    def __init__(self, csv_file, root_dir, augment=False, mmap=False, img_dir="./Data/gta_data/"):

        self.data = pd.read_csv(root_dir + csv_file, index_col=0, sep=",")
        #self.data = quantized_stat(self.data)
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.augment = augment
        self.mmap = mmap
        self.preprocess = PREPROCESS()
        
        self.data["path"] = self.data["path"].str.split(pat="/", n = 3, expand=True).iloc[:, -1]
    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img_names = self.img_dir + self.data.iloc[idx, 4]
    
        if(isinstance(img_names, str)):
            img_names = [img_names]
            
        if(self.mmap):
            mmaps = list()
            
            
        images = list()
        
        for im_name in img_names:
            
            image = io.imread(im_name)
        
            if(self.mmap):
                mmap = self.preprocess.preprocess_mmap(image)
                mmaps.append(mmap)
        
        
            if self.augment:
                image = self.preprocess.preprocess_augment_image(image)
            else:
                image = self.preprocess.preprocess_image(image)
        
            images.append(image)
        
        if(len(img_names)>1):
            images = [el.unsqueeze(0) for el in images]
            if(self.mmaps):
                mmaps = [el.unsqueeze(0) for el in mmaps]
            
        images = torch.cat(images)
        
        if(self.mmap):
            mmaps = torch.cat(mmaps)
        
        statistics = self.data.iloc[idx, :2]
        statistics = np.array(statistics, dtype=np.float32)
        statistics = torch.tensor(statistics, dtype=torch.float32)
        
        speed = self.data.iloc[idx, 3]/self.data["speed"].max()
        speed = np.array(speed, dtype=np.float32)
        speed = torch.tensor(speed, dtype=torch.float32)
    

        if(self.mmap):
            sample = {'img': images, 'mmap': mmaps, 'speed': speed, 'statistics': statistics}
            return sample
            
            
        sample = {'img': images, 'speed': speed, 'statistics': statistics}

        return sample
    

#============================================================================
#====================== Sørensen–Dice coefficient ===========================
#============================================================================
    
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)
    
    
    
#============================================================================
#====================== Segmentation dataset ===========================
#============================================================================

   

class GTA_segment_Dataset(Dataset):

    def __init__(self, csv_file, root_dir, img_dir=""):

        self.data_csv = pd.read_csv(root_dir + csv_file, index_col=0)
        self.root_dir = root_dir
        self.img_dir = img_dir
        self.preprocess = PREPROCESS()
    

    def __len__(self):
        return len(self.data_csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        
        img_names = self.root_dir + self.img_dir + self.data_csv.iloc[idx, 0]
    
        if(isinstance(img_names, str)):
            img_names = [img_names]
            
            
        images = list()
        
        for im_name in img_names:
            
            image = io.imread(im_name)
        
            image = self.preprocess.preprocess_segment(image)
        
            images.append(image)
        
        if(len(img_names)>1):
            images = [el.unsqueeze(0) for el in images]
            
        images = torch.cat(images)
        
        
        mask_names = self.root_dir + self.img_dir + self.data_csv.iloc[idx, 1]
        
        if(isinstance(mask_names, str)):
            mask_names = [mask_names]
            
            
        masks = list()
        
        for mask_name in mask_names:
            
            mask = np.array(Image.open(mask_name))
        
            mask = self.preprocess.preprocess_mask(mask)

            masks.append(mask)
        
        if(len(mask_names)>1):
            masks = [el.unsqueeze(0) for el in masks]
            
        masks = torch.cat(masks)
    
            
            
        sample = {'img': images, 'mask': masks}

        return sample
    
    
    
    