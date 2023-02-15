from libraries import *
import baseFunctions as bf




class ViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.d_model = d_model
        
        self.conv_proj = nn.Conv2d(in_channels=3, out_channels=d_model, kernel_size=patch_size, stride=patch_size)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward = 1024, norm_first = True, batch_first = True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
    
    def _process_input(self, x, x_speed):
        n, c, h, w = x.shape
        
        torch._assert(h == self.image_size, f"Wrong image height! Expected {self.image_size} but got {h}!")
        torch._assert(w == self.image_size, f"Wrong image width! Expected {self.image_size} but got {w}!")
        
        n_h = h // self.patch_size
        n_w = w // self.patch_size
        
        # (B, C, H, W) -> (B, d_model, n_h, n_w)
        x = self.conv_proj(x)
        
        # (B, d_model, n_h, n_w) -> (B, d_model, (n_h * n_w))
        x = x.reshape(n, self.d_model, n_h * n_w)
        
        x = x.permute(0, 2, 1)  #(B, S, D)
        
        #(B,1) -> (B, 1, D)
        x_speed = x_speed.expand(-1, self.d_model).unsqueeze(1)
        
        x = torch.cat([x, x_speed], axis = 1) #(B, S+1, D)
        
        return x
    
    def forward(self, x, x_speed):
        x = self._process_input(x, x_speed)
        
        return self.transformer_encoder(x)