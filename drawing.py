from tkinter import *
from tkinter import ttk, colorchooser, filedialog
import PIL
from PIL import ImageGrab
# from PIL import Image
from PIL import Image, ImageDraw
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm.notebook import tqdm
from torchvision import datasets, transforms
import os
import cv2
import numpy as np
import torch
from matplotlib.pyplot import imshow
from IPython.display import display
from PIL import Image
from torch.autograd import Variable
from models import create_model
from options.train_options import TrainOptions
from options.base_options import *
from options.test_options import *
from models.pix2pix_model import *
from data import *
import matplotlib.pyplot as plt
import PIL

class main:
    def __init__(self,master):
        self.master = master
        self.color_fg = 'black'
        self.color_bg = self._from_rgb((255, 255, 255))
        self.old_x = None
        self.old_y = None
        self.penwidth = 5
        self.drawWidgets()
        self.c.bind('<B1-Motion>',self.paint)
        self.c.bind('<ButtonRelease-1>',self.reset)

    def paint(self,e):
        if self.old_x and self.old_y:
            self.c.create_line(self.old_x,self.old_y,e.x,e.y,width=self.penwidth,fill=self.color_fg,capstyle=ROUND,smooth=True)
            # print(int(float(self.penwidth)))
            w = int(float(self.penwidth))
            num = 2
            self.draw.line([self.old_x,self.old_y,e.x,e.y], self.picker, width=w)
            self.draw.ellipse([self.old_x-w//num,self.old_y-w//num,e.x+w//num,e.y+w//num], fill=self.picker)

        self.old_x = e.x
        self.old_y = e.y

    def reset(self,e):
        self.old_x = None
        self.old_y = None      

    def changeW(self,e):
        self.penwidth = e



    def clear(self):
        self.c.delete(ALL)
        self.image1 = Image.new("RGB", (256, 256), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)

    def change_fg(self):
        self.color_fg=colorchooser.askcolor(color=self.color_fg)[1]

    def change_bg(self):
        self.color_bg=colorchooser.askcolor(color=self.color_bg)[1]
        self.c['bg'] = self.color_bg

    def set_clouds(self):
        self.color_fg = self._from_rgb((255, 255, 255))
        self.picker = (255, 255, 255)
    
    def set_grass(self):
        self.color_fg = self._from_rgb((0, 255, 0))
        self.picker = (0, 255, 0)
    
    def set_sky(self):
        self.color_fg = self._from_rgb((0, 197, 229))
        self.picker = (0, 197, 229)

    def set_tree(self):
        self.color_fg = self._from_rgb((0, 154, 78))
        self.picker = (0, 154, 78)

    def set_dirt(self):
        self.color_fg = self._from_rgb((143, 74, 0))
        self.picker = (143, 74, 0)
    
    def set_water(self):
        self.color_fg = self._from_rgb((0, 64, 143))
        self.picker = (0, 64, 143)

    def save(self):
        # self.image1.show()
        img = self.image_loader(self.image1)

        self.model.eval()
        self.model.real_A = img
        self.model.forward()
        output = self.model.fake_B
        
        plt.imshow(self.tensor2im(output))
        plt.axis('off')
        plt.show() 


    def _from_rgb(self, rgb):
        """translates an rgb tuple of int to a tkinter friendly color code
        """
        return "#%02x%02x%02x" % rgb  

    def image_loader(self, image_name):
        """load image, returns cuda tensor"""
        print(type(image_name))
        # image = Image.fromarray(cv2.cvtColor(image_name, cv2.COLOR_BGR2RGB).astype('uint8'), 'RGB')
        # image = Image.fromarray(image_name, 'RGB')
        # image_name.show()
        image = image_name
        image = self.loader(image).float()
        image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
        return image.cuda()  #assumes that you're using GPU

    def tensor2im(self, input_image, imtype=np.uint8):
        """"Converts a Tensor array into a numpy image array.

        Parameters:
            input_image (tensor) --  the input image tensor array
            imtype (type)        --  the desired type of the converted numpy array
        """
        if not isinstance(input_image, np.ndarray):
            if isinstance(input_image, torch.Tensor):  # get the data from a variable
                image_tensor = input_image.data
            else:
                return input_image
            image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
            if image_numpy.shape[0] == 1:  # grayscale to RGB
                image_numpy = np.tile(image_numpy, (3, 1, 1))
            image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image
        return image_numpy.astype(imtype)

    def drawWidgets(self):
        self.controls = Frame(self.master,padx = 5,pady = 5)
        Label(self.controls, text='Pen Width: ',font=('',15)).grid(row=0,column=0)
        self.slider = ttk.Scale(self.controls,from_= 5, to = 100, command=self.changeW,orient=HORIZONTAL)
        self.slider.set(self.penwidth)
        self.slider.grid(row=0,column=1,ipadx=30)
        self.controls.pack()
        
        self.c = Canvas(self.master,width=256,height=256,bg=self.color_bg,)
        self.c.pack(fill=BOTH,expand=True)
        
        self.image1 = Image.new("RGB", (256, 256), (255, 255, 255))
        self.draw = ImageDraw.Draw(self.image1)
        self.set_dirt()

        menu = Menu(self.master)
        self.master.config(menu=menu)

        filemenu = Menu(menu)
        menu.add_cascade(label='Convert',menu=filemenu)
        filemenu.add_command(label='Convert',command=self.save)

        # colormenu = Menu(menu)
        # menu.add_cascade(label='Colors',menu=colormenu)
        # colormenu.add_command(label='Brush Color',command=self.change_fg)
        # colormenu.add_command(label='Background Color',command=self.change_bg)

        optionmenu = Menu(menu)
        menu.add_cascade(label='Options',menu=optionmenu)
        optionmenu.add_command(label='Clear Canvas',command=self.clear)
        optionmenu.add_command(label='Exit',command=self.master.destroy) 

        picker = Menu(menu)
        menu.add_cascade(label='ColorPicker',menu=picker)
        picker.add_command(label='Clouds',command=self.set_clouds)
        picker.add_command(label='Grass', command=self.set_grass)
        picker.add_command(label='Sky',command=self.set_sky)
        picker.add_command(label='Tree', command=self.set_tree)
        picker.add_command(label='Dirt',command=self.set_dirt)
        picker.add_command(label='Water', command=self.set_water)
        
        ###
        opt = TestOptions()
        opt.gpu_ids = [0]
        opt.isTrain = False
        opt.checkpoints_dir = "C:\\Users\\oranl\\Desktop\\draw_gen\\checkpoints"
        opt.name = "pix2pixTester"
        opt.preprocess = "resize_and_crop"
        opt.input_nc = 3
        opt.output_nc = 3
        opt.ngf = 64
        opt.netG = "unet_256"
        opt.norm = "batch"
        opt.no_dropout = "store_true"
        opt.init_type = "normal"
        opt.init_gain = .02
        opt.model="pix2pix"
        opt.direction = "AtoB"

        self.model = Pix2PixModel(opt)
        self.model.load_networks("latest")
        def __crop(img, pos, size):
            ow, oh = img.size
            x1, y1 = pos
            tw = th = size
            if (ow > tw or oh > th):
                return img.crop((x1, y1, x1 + tw, y1 + th))

            return img

        self.loader = transforms.Compose([transforms.Resize(size=[256, 256], interpolation=PIL.Image.BICUBIC), 
                             transforms.Lambda(lambda img: __crop(img, (0, 0), 256)),
                            transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        

if __name__ == '__main__':
    root = Tk()
    main(root)
    root.title('DrawingApp')
    root.mainloop()