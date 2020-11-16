import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2 as cv
from copy import deepcopy
from mpl_toolkits import mplot3d
import random


numbers=['1','2','3']
def displayChannel():
 for x in numbers:
  img=mpimg.imread('0'+x+'.jpg')

  fig=plt.figure(figsize=(15,5))
  ax=fig.add_subplot(1,4,1)
  pic=plt.imshow(img[:,:,0],cmap='Reds_r')
  ax.set_title('Red'),plt.xticks([]), plt.yticks([])

  ax=fig.add_subplot(1,4,2)
  pic=plt.imshow(img[:,:,0],cmap='Greens_r')
  ax.set_title('Green'),plt.xticks([]), plt.yticks([])

  ax=fig.add_subplot(1,4,3)
  pic=plt.imshow(img[:,:,0],cmap='Blues_r')
  ax.set_title('Blue'),plt.xticks([]), plt.yticks([])

  ax=fig.add_subplot(1,4,4)
  pic=plt.imshow(img[:,:,0],cmap='Greys_r')
  ax.set_title('Grey'),plt.xticks([]), plt.yticks([])
  plt.show(pic)


def rgbExclusion(Ch):
      
      img=mpimg.imread('02.jpg')

      fig=plt.figure(figsize=(15,5))
      
      if Ch=='r':
          ax=fig.add_subplot(1,2,1)
          pic=plt.imshow(img[:,:,0],cmap='Greens_r')
          ax.set_title('Green'),plt.xticks([]), plt.yticks([])

          ax=fig.add_subplot(1,2,2)
          pic=plt.imshow(img[:,:,0],cmap='Blues_r')
          ax.set_title('Blue'),plt.xticks([]), plt.yticks([])

          
          plt.show(pic)
          
      if Ch=='g':
          ax=fig.add_subplot(1,2,1)
          pic=plt.imshow(img[:,:,0],cmap='Reds_r')
          ax.set_title('Red'),plt.xticks([]), plt.yticks([])

          ax=fig.add_subplot(1,2,2)
          pic=plt.imshow(img[:,:,0],cmap='Blues_r')
          ax.set_title('Blue'),plt.xticks([]), plt.yticks([])

          
          plt.show(pic)
          
      if Ch=='b':
          ax=fig.add_subplot(1,2,1)
          pic=plt.imshow(img[:,:,0],cmap='Greens_r')
          ax.set_title('Green'),plt.xticks([]), plt.yticks([])

          ax=fig.add_subplot(1,2,2)
          pic=plt.imshow(img[:,:,0],cmap='Reds_r')
          ax.set_title('Red'),plt.xticks([]), plt.yticks([])

          
          plt.show(pic)   


def displayHist():
 
  for x in numbers:
     img=mpimg.imread('0'+x+'.jpg')
     fig=plt.figure(figsize=(15,3))
     
     ax=fig.add_subplot(1,4,2)
     plt.hist(img.ravel(),bins=256)
     ax.set_title('Historgram'),plt.xticks([]), plt.yticks([])
     
     ax=fig.add_subplot(1,4,1)
     plt.imshow(img[:,:,0],cmap='Greys_r')
     ax.set_title('Greyscale'),plt.xticks([]), plt.yticks([])
     
     ax=fig.add_subplot(1,4,3)
     img1=cv.imread('0'+x+'.jpg',0)
     equ = cv.equalizeHist(img1)
     plt.imshow(equ,cmap='Greys_r')
     ax.set_title('After HE Image'),plt.xticks([]), plt.yticks([])
     
     ax=fig.add_subplot(1,4,4)
     plt.hist(equ.ravel(),bins=256)
     ax.set_title('After HE Histogram'),plt.xticks([]), plt.yticks([])
     
     
def convOp(image,Kernel):
    ori_image=cv.imread(image)
    ori_image = cv.cvtColor(ori_image, cv.COLOR_BGR2GRAY)
    blur2 = cv.GaussianBlur(ori_image,(5,5),4)
  
    
    kernelSharp=np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
   # print(type(kernelSharp))
   
    kernelBlur=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0
    
    if Kernel=='s':
       kernel=np.flipud(np.fliplr(kernelSharp))
    
    
       output= np.zeros_like(ori_image)
    
       image_padded=np.zeros((ori_image.shape[0]+2,ori_image.shape[1]+2))
       image_padded[1:-1,1:-1]=ori_image
    
       for x in range(ori_image.shape[1]):
           for y in range(ori_image.shape[0]):
               output[y,x]=(kernel * image_padded[y: y+3, x: x+3]).sum()
      
        
       fig=plt.figure(figsize=(15,5))
       ax=fig.add_subplot(1,2,1)
       plt.imshow(ori_image,cmap='Greys_r')
       ax.set_title('Before Conv'),plt.xticks([]), plt.yticks([])   
       ax=fig.add_subplot(1,2,2)
       plt.imshow(output,cmap='Greys_r')
       ax.set_title('After Conv (Sharp)'),plt.xticks([]), plt.yticks([]) #there is no Built in function for sharping
    
    
    if Kernel=='b':
       kernel=np.flipud(np.fliplr(kernelBlur))
    
    
       output= np.zeros_like(ori_image)
    
       image_padded=np.zeros((ori_image.shape[0]+2,ori_image.shape[1]+2))
       image_padded[1:-1,1:-1]=ori_image
    
       for x in range(ori_image.shape[1]):
           for y in range(ori_image.shape[0]):
               output[y,x]=(kernel * image_padded[y: y+3, x: x+3]).sum()
      
        
       fig=plt.figure(figsize=(15,5))
       ax=fig.add_subplot(1,3,1)
       plt.imshow(ori_image,cmap='Greys_r')
       ax.set_title('Before Conv'),plt.xticks([]), plt.yticks([])    
       ax=fig.add_subplot(1,3,2)
       plt.imshow(output,cmap='Greys_r')
       ax.set_title('After Conv (Blur)'),plt.xticks([]), plt.yticks([])
       ax=fig.add_subplot(1,3,3)
       plt.imshow(blur2,cmap='Greys_r')
       ax.set_title('Built in (Blur)'),plt.xticks([]), plt.yticks([])
    
    
     


def convBox(image, kernel):
       ori_image=cv.imread(image)
       ori_image = cv.cvtColor(ori_image, cv.COLOR_BGR2GRAY)
  
    
 
       kernel=np.flipud(np.fliplr(kernel))
    
    
       output= np.zeros_like(ori_image)
    
       image_padded=np.zeros((ori_image.shape[0]+2,ori_image.shape[1]+2))
       image_padded[1:-1,1:-1]=ori_image
    
       for x in range(ori_image.shape[1]):
           for y in range(ori_image.shape[0]):
               output[y,x]=(kernel * image_padded[y: y+3, x: x+3]).sum()
      
        
       fig=plt.figure(figsize=(15,5))
       ax=fig.add_subplot(1,2,1)
       plt.imshow(ori_image,cmap='Greys_r')
       ax.set_title('Before Conv'),plt.xticks([]), plt.yticks([])   
       ax=fig.add_subplot(1,2,2)
       plt.imshow(output,cmap='Greys_r')
       ax.set_title('After Conv'),plt.xticks([]), plt.yticks([])


def gaussianFil(image):
    image = cv.imread(image)
   
    blur2 = cv.GaussianBlur(image,(5,5),4)
    blur3 = cv.GaussianBlur(image,(7,7),16)
    blur4 = cv.GaussianBlur(image,(11,11),64)
    
    fig=plt.figure(figsize=(25,5))
    
    ax=fig.add_subplot(141),plt.imshow(image),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(142),plt.imshow(blur2),plt.title('Blurred with sigma=4')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(143),plt.imshow(blur3),plt.title('Blurred with sigma=16')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(144),plt.imshow(blur4),plt.title('Blurred with sigma=64')
    plt.xticks([]), plt.yticks([])
    plt.show()
    
def noise(image1):
      
    image=cv.imread(image1,cv.IMREAD_GRAYSCALE)
    fig=plt.figure(figsize=(15,5))
    
    ax=fig.add_subplot(131),plt.imshow(image,cmap='Greys_r'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    gaussianNoise=np.zeros((image.shape[0],image.shape[1]),dtype=np.uint8)
    cv.randn(gaussianNoise, 128, 20)
    cv.waitKey()
    
    
    
    
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - 0.05 
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < 0.05:
                output[i][j] = 0
            elif rdn > thres:
                output[i][j] = 255
            else:
                output[i][j] = image[i][j]
    

    cv.imwrite('sp_noise.jpg', output)
    
    ax=fig.add_subplot(132),plt.imshow(output,cmap='Greys_r'),plt.title('Salt and Pepper Noise')
    plt.xticks([]), plt.yticks([])
    

    gaussianNoise = (gaussianNoise*0.5).astype(np.uint8)
    noisy_image1 = cv.add(image,gaussianNoise)
    cv.imwrite('co.jpg',noisy_image1)

    ax=fig.add_subplot(133),plt.imshow(noisy_image1,cmap='Greys_r'),plt.title('Gaussian Nosie')
    plt.xticks([]), plt.yticks([])
    
    
def gaussMedFil(image):
    image = cv.imread(image,cv.IMREAD_GRAYSCALE)
    img=cv.imread('co.jpg')
    img1=cv.imread('sp_noise.jpg')
    blur2 = cv.GaussianBlur(img,(5,5),1)
    blur3 = cv.medianBlur(img1,11)

    
    fig=plt.figure(figsize=(40,30))
    
    ax=fig.add_subplot(261),plt.imshow(image,cmap='Greys_r'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(262),plt.imshow(img),plt.title('Gaussian Noise')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(263),plt.imshow(blur2,cmap='Greys_r'),plt.title('Gaussian Filter')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(161),plt.imshow(image,cmap='Greys_r'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(162),plt.imshow(img1,cmap='Greys_r'),plt.title('Salt and Pepper Noise')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(163),plt.imshow(blur3,cmap='Greys_r'),plt.title('Median Filter')
    plt.xticks([]), plt.yticks([])


def meshPlotGaussian():
    n_points = 40
    x_vals = np.arange(n_points)
    y_vals = np.random.normal(size=n_points)
    x = np.arange(-6, 6, 0.1) # x from -6 to 6 in steps of 0.1
    y = 1 / np.sqrt(2 * np.pi) * np.exp(-x ** 2 / 2.)
   
  
    
    
    sigma = 2
    x_position = 13 # 14th point
    kernel_at_pos = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2)) 
    kernel_at_pos = kernel_at_pos / sum(kernel_at_pos)
    
    y_by_weight = y_vals * kernel_at_pos # element-wise multiplication
    new_val = sum(y_by_weight)
    
    smoothed_vals = np.zeros(y_vals.shape)
    for x_position in x_vals:
        kernel = np.exp(-(x_vals - x_position) ** 2 / (2 * sigma ** 2))
        kernel = kernel / sum(kernel)
        smoothed_vals[x_position] = sum(y_vals * kernel)
    
    fig=plt.figure(figsize=(15,5))
    ax = fig.add_subplot(111, projection='3d')
    dx = 0.1
    dy = 0.1
    x = np.arange(-6, 6, dx)
    y = np.arange(-6, 6, dy)
    x2d, y2d = np.meshgrid(x, y)
    kernel_2d = np.exp(-(x2d ** 2 + y2d ** 2) / (2 * sigma ** 2))
    kernel_2d = kernel_2d / (2 * np.pi * sigma ** 2) # unit integral
    ax.plot_surface(x2d, y2d,kernel_2d),plt.title('Gaussian Filter with sigma value='+str(sigma))
    


 
def sobelOp(image):
    img=cv.imread(image,0)
    


  
    sobelx = cv.Sobel(img,cv.CV_64F,1,0,ksize=5)
    sobely = cv.Sobel(img,cv.CV_64F,0,1,ksize=5)
    
    combineSobel=cv.addWeighted(sobelx,0.5,sobely,0.5,0)
    
    gradient_magnitude = np.sqrt(np.square(sobelx) + np.square(sobely))
 
    gradient_magnitude *= 255.0 / gradient_magnitude.max()

    fig=plt.figure(figsize=(15,5))
    
    ax=fig.add_subplot(131),plt.imshow(img,cmap='Greys_r'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(132),plt.imshow(combineSobel,cmap='Greys_r'),plt.title('Sobel Image')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(133),plt.imshow(gradient_magnitude,cmap='Greys_r'),plt.title('Gradient Magnitude Image')
    plt.xticks([]), plt.yticks([])

    plt.show()
    
    
def Laplacian(image):
    img=cv.imread(image,0)
    laplacian = cv.Laplacian(img,cv.CV_64F) 
    gradient_magnitude = np.sqrt(np.square(laplacian))
 
    gradient_magnitude *= 255.0 / gradient_magnitude.max()
    
    fig=plt.figure(figsize=(15,5))
    ax=fig.add_subplot(131),plt.imshow(img,cmap='Greys_r'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(132),plt.imshow(laplacian,cmap='Greys_r'),plt.title('Laplacian Image')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(133),plt.imshow(gradient_magnitude,cmap='Greys_r'),plt.title('Laplacian Magnitude Image')
    plt.xticks([]), plt.yticks([])
    
def canneyEdge(image):
    img=cv.imread(image,0)
    edges = cv.Canny(img,100,200)
    
    fig=plt.figure(figsize=(15,5))
    ax=fig.add_subplot(131),plt.imshow(img,cmap='Greys_r'),plt.title('Original')
    plt.xticks([]), plt.yticks([])
    ax=fig.add_subplot(132),plt.imshow(edges,cmap='Greys_r'),plt.title('Edge Detection Image')
    plt.xticks([]), plt.yticks([])

def canneyVideo():
    cap = cv.VideoCapture(0)
    while True:
     ret, img = cap.read()
    
     gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     blur = cv.GaussianBlur(gray, (5, 5), 0)
     canny = cv.Canny(blur, 10, 70)
     ret, mask = cv.threshold(canny, 70, 255, cv.THRESH_BINARY)
     cv.imshow('Video feed', mask)
    
     if cv.waitKey(1) == 13:
          break
         
    cap.release()
    cv.destroyAllWindows()
    