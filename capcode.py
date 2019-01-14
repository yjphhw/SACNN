#!/usr/bin/env python 
#coding:utf8
import random
import string
import sys
import glob
import math
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class VerificationCode():
    def __init__(self):
        # 字体的位置，不同版本的系统会有不同
        self.font_paths=glob.glob(r'C:\Windows\Fonts\*.ttf')
        
        # 生成几位数的验证码
        self.number = 1
        # 生成验证码图片的高度和宽度
        #self.size = (136, 32)
        self.size = (136, 136)
        self.font_size=(30,32)
        # 背景颜色，默认为白色
        self.bgcolor = (255, 255, 255)
        # 字体颜色，默认为蓝色
        self.fontcolor = self.get_random_color()
        # 干扰线颜色。随机设置
        self.linecolor = self.get_random_color()
        # 是否要加入干扰线
        self.draw_line = True
        # 加入干扰线条数的上下限
        self.line_number = 3
        #验证码内容
        self.text = ""
    def get_random_color(self,low=0,high=230):
        return random.randint(low,high), random.randint(low,high), random.randint(low,high)
    # 用来随机生成一个字符串
    def gene_text(self):
        source = list(string.ascii_letters)
        for index in range(0, 10):
            source.append(str(index))
        self.text = "".join(random.sample(source, self.number))  # number是生成验证码的位数
        return self.text
    # 用来绘制干扰线
    def gene_line(self,draw, width, height):
        begin = (random.randint(0, width), random.randint(0, height))
        end = (random.randint(0, width), random.randint(0, height))
        draw.line([begin, end], fill=self.linecolor)

    # 生成验证码
    def gene_code(self,ISNULL=False,ISOUTIMG=False):
        width, height = self.size  # 宽和高
        image = Image.new('RGBA', (width, height), self.bgcolor)  # 创建图片
        font_path=random.choice(self.font_paths)
        
        draw = ImageDraw.Draw(image)  # 创建画笔
        if not ISNULL:
            text = self.gene_text()  # 生成字符串
            font = ImageFont.truetype(font_path, random.randint(*self.font_size))  # 验证码的字体
            font_width, font_height = font.getsize(text)
            dx=random.choice(range(24))-12
            dy=random.choice(range(24))-12
            draw.text(((width - font_width) / 2+dx, (height - font_height) / 2+dy), text, font=font, fill=self.get_random_color())  # 填充字符串
        
        else:
            font = ImageFont.truetype(font_path, int(random.randint(*self.font_size)*random.choice([0.1,4,6])))  # 验证码的字体
            text=self.gene_text() if random.random()>0.5 else ''
            font_width, font_height = font.getsize(text)
            dx=random.choice(range(24))-12
            dy=random.choice(range(24))-12
            draw.text(((width - font_width) / 2+dx, (height - font_height) / 2+dy), text, font=font, fill=self.get_random_color())  # 填充字符串
        
        #print(text)
        
        #print(font_width,font_height)
        
        if self.draw_line:
            for i in range(self.line_number):
                self.gene_line(draw, width, height)

        #simage = image.transform((width, height), Image.AFFINE, (1, -0.3, 0, -0.1, 1, 0), Image.BILINEAR)  # 创建扭曲
        image=image.convert('L')
        if ISOUTIMG:
            image = image.filter(ImageFilter.EDGE_ENHANCE_MORE)  # 滤镜，边界加强
            image.save("./tmp/{0}.png".format(self.text),"png")  # 保存验证码图片

        return text,image
 
class CapchaDataset(Dataset):
    "capcha dataset ."
    def __init__(self,ratio=1,txtnum=1,v=VerificationCode(),transform=None):
        self.v=v
        self.v.number=txtnum
        self.transform=transform
        self.ratio=ratio
    def __len__(self):
        return 100000
    def __getitem__(self,idx):
        sample={}
        if random.choice([1]+[0]*self.ratio):
            sample['image']=self.v.gene_code(ISNULL=False)[1]
            sample['landmarks']=torch.Tensor([1])
        else:
            sample['image']=self.v.gene_code(ISNULL=True)[1]
            sample['landmarks']=torch.Tensor([0])
        if self.transform:
            sample['image']=self.transform(sample['image'])
        return sample['image'],sample['landmarks']


def main():
    vc = VerificationCode()
    vc.number=1
    #vc.gene_text()
    for i in range(5000):
        t,img=vc.gene_code(ISNULL=i%2,ISOUTIMG=False)
        img.save("./cd/{}/{}.png".format(i%2,i//2))
    

if __name__ == "__main__":
    main()
