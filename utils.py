import lpips
import numpy as np
import torchvision.transforms as T

def compute_lpips_diversity(images):
   loss_fn = lpips.LPIPS(net='alex')  # or use 'vgg'
   to_tensor = T.ToTensor()

   # Convert PIL to normalized tensors
   tensors = [to_tensor(img).unsqueeze(0) for img in images]
   scores = []

   for i in range(len(tensors)):
       for j in range(i + 1, len(tensors)):
           d = loss_fn(tensors[i], tensors[j])
           scores.append(d.item())

   return np.mean(scores)