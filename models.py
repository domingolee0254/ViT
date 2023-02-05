import torch
import torch.nn as nn

# Params
#   img_tensor: 4D torch Tensor (batch_size, channel, height, width)
# 3 x 32 x 32 -(embedding)-> emb_dim
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=768):
        super(PatchEmbedding, self).__init__()
        self.embedding = nn.Conv2d(in_channels=3, 
                                   out_channels=emb_dim, 
                                   kernel_size=patch_size,
                                   stride=patch_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_dim))
        self.positions = nn.Parameter(torch.randn((img_size // patch_size) **2 + 1, emb_dim))

        # %JYP pos_embedding
     
    def forward(self, x):
        x = self.embedding(x)
        batch_size, emb_dim, _, _ = x.size()
        x = x.view(batch_size, emb_dim, -1) 
        x = x.permute(0,2,1)                                    #[batch_size, 196, emb_dim]
        cls_token = self.cls_token.repeat(batch_size, 1, 1)     #[batch_size, 1, emb_dim]
        x = torch.cat((cls_token, x), dim=1)
        x += self.positions

        return x

class MSA(nn.module):
    def __init__(self):
        super(MSA, self, q, k, v).__init__()
        self.q = q
        self.k = k
        self.v = v
        pass

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, emb_dim=768):
        super(VisionTransformer, self).__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size, emb_dim)
        #self.encoder = MSA()
        #self.MLP = nn.Linear(emb_dim, num_classes)
    def forward(self, x):
        x = self.patch_embedding(x) # (batch_size, emb_dim, dim2, dim3)

        return x.shape

if __name__ == "__main__":
    x = torch.rand(1,3,224,224)
    tmp = VisionTransformer(img_size=224, patch_size=16, emb_dim=768)
    print(tmp.forward(x))