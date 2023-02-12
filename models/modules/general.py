from libraries import *
import baseFunctions as bf


class CONV_BLOCK(nn.Module):
     def __init__(self, in_dim, out_dim, kernel, stride, padding, pool):
         super().__init__()
         
         self.conv = nn.Conv2d(in_dim, out_dim, kernel, stride, padding)
         self.batchNorm = nn.BatchNorm2d(out_dim)
         self.relu = nn.ReLU()
         
         self.pool = nn.MaxPool2d(pool, pool)
         
     def forward(self, x):
         x = self.relu(self.batchNorm(self.conv(x)))
         return self.pool(x)


class RES_BLOCK(nn.Module):
    def __init__(self, in_dim, out_dim, kernel, num_conv, pool):
        super().__init__()
        
        layers = list()
        
        for i in range(num_conv):
            d_in = in_dim if i == 0 else out_dim
            layers.append(nn.Conv2d(d_in, out_dim, kernel, 1, "same"))
            layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.ReLU())
            
        self.net = nn.Sequential(*layers)
        
        self.finalConv2d = nn.Conv2d(out_dim, out_dim, kernel, 1, "same")
        self.finalBN = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU()
        
        
        self.conv11 = nn.Conv2d(in_dim, out_dim, 1, 1, "same")
        
        self.pool = nn.MaxPool2d(pool, pool)
        
    def forward(self, x):
        h = self.net(x)
        
        h = self.finalBN(self.finalConv2d(h))
        
        x = self.relu(h+self.conv11(x))
        
        return self.pool(x)

class MMAP_CONV(nn.Module):
    def __init__(self):
        super().__init__()
        
        #INPUT = 40x40x3
        
        self.conv1_map = nn.Conv2d(3, 64, 5, 1, 0) # 36X36X64
        self.batchNorm1_map = nn.BatchNorm2d(64)
        self.convlRelu1_map = nn.ReLU() 
        self.maxPool_1 = nn.MaxPool2d(2) # 18X18X64
        
        self.conv2_map = nn.Conv2d(64, 128, 3, 1, 0) # 16X16X128
        self.batchNorm2_map = nn.BatchNorm2d(128)
        self.convlRelu2_map = nn.ReLU() 
        self.maxPool_2 = nn.MaxPool2d(2) # 8X8X128
        
        self.flat = nn.Flatten() #8192
        
    def forward(self, x):
        
        x = self.maxPool_1(self.convlRelu1_map(self.batchNorm1_map(self.conv1_map(x))))
        x = self.maxPool_2(self.convlRelu2_map(self.batchNorm2_map(self.conv2_map(x))))
        return self.flat(x)
    
    
class NormLinear(nn.Module):
    def __init__(self, in_dim, out_dim, p):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(p=p)
        
    def forward(self, x):
        x = self.drop(self.relu(self.bn(self.linear(x))))
        return x
        
    
class MLP(nn.Module):
    
    def __init__(self, in_dim, hidden_dims = [1024, 256, 64]):
        super().__init__()
        
        layers = list()
        
        for i in range(len(hidden_dims)):
            
            d_in = in_dim if i == 0 else hidden_dims[i-1]
            
            layers.append(NormLinear(d_in, hidden_dims[i], 0.1))
        
        self.net = nn.Sequential(*layers)


    def forward(self, x):
        return self.net(x)
        