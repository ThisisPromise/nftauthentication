import cv2
import numpy as np 
from skimage.metrics import structural_similarity as ssim 
import matplotlib.pyplot as plt 



imageA = cv2.imread("abstract_image1.png", cv2.IMREAD_GRAYSCALE)
imageB = cv2.imread("abstract_image2.png", cv2.IMREAD_GRAYSCALE)

#compute ssim
ssim_index, ssim_map = ssim(imageA,imageB, full= True)

f_transform_imgA = np.fft.fft2(imageA)
f_transform_imgB = np.fft.fft2(imageB)

f_shift1 = np.fft.fftshift(f_transform_imgA)
f_shift2 = np.fft.fftshift(f_transform_imgB)

magnitude_spectrum1 = np.log(np.abs(f_shift1) + 1)
magnitude_spectrum2 = np.log(np.abs(f_shift2) + 1)

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(imageA, cmap='gray')
plt.title('Image 1')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(imageB, cmap='gray')
plt.title('Image 2')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(ssim_map, cmap='hot')
plt.title('SSIM Map')
plt.colorbar()
plt.axis('off')

plt.show()

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(magnitude_spectrum1, cmap='gray')
plt.title('Magnitude Spectrum Image 1')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum2, cmap='gray')
plt.title('Magnitude Spectrum Image 2')
plt.axis('off')

plt.show()

print("SSIM Index:", ssim_index)