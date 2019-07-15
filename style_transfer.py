import torch
import numpy as np
import matplotlib.pyplot as plt

import model
import data
import cv2
import random

cinn = model.MNIST_cINN(0)
cinn.cuda()
state_dict = {k:v for k,v in torch.load('output/mnist_cinn.pt').items() if 'tmp_var' not in k}
cinn.load_state_dict(state_dict)

cinn.eval()

def val_loss():
    '''prints the final validiation loss of the model'''
    random_l = data.val_l.cpu().numpy()
    random.shuffle(random_l)
    random_l = torch.Tensor(random_l).long().cuda()

    with torch.no_grad():
        z, _ = cinn(data.val_x, data.val_l)

        recon = cinn.reverse_sample(z,data.val_l).cpu().numpy()

        fake = cinn.reverse_sample(z,random_l).cpu().numpy()

    nrow =20
    ncol =30
    n_imgs = 3
    full_image = np.zeros((28*nrow, 28*ncol*n_imgs))

    for s in range(nrow*ncol):
        i, j = s // ncol , s % ncol
        # Show src Image
        src_show = data.val_x[s, 0].cpu().numpy()
        src_show = data.unnormalize(src_show)
        full_image[28 * i : 28 * (i+1), 28 * j*n_imgs : 28 * (j*n_imgs+1)] = src_show

        # Show recon Image
        rec_show = recon[s, 0]
        rec_show = data.unnormalize(rec_show)
        full_image[28 * i : 28 * (i+1), 28 * (j*n_imgs+1) : 28 * (j*n_imgs+2)] = 1 - rec_show

        # Show random label Image
        fake_show = fake[s, 0]
        fake_show = data.unnormalize(fake_show)
        full_image[28 * i : 28 * (i+1), 28 * (j*n_imgs+2) : 28 * (j*n_imgs+3)] = 1 - fake_show

    
    full_image = np.clip(full_image, 0 , 1)
    plt.title(F'Left: val source image ; Mid: Recon Image Right: Random Label image')
    plt.imshow(full_image, vmin=0, vmax=1, cmap='gray')
    cv2.imwrite("source-vs-rec.png",full_image*255.0)


val_loss()

plt.show()

