import numpy as np

SE = np.array([[1, 1, 1],
               [1, 1, 1],
               [1, 1, 1]])

def erosion(img, SE=SE):
  # The padding of the rows and columns should depend on the SE.
  # Padding of 2 is used specifically when the SE is 3x3.
  rows_padding=SE.shape[0]//2
  cols_padding=SE.shape[1]//2
  img_padding=np.zeros(shape=(img.shape[0]+2*rows_padding, img.shape[1]+2*cols_padding))
  img_padding[rows_padding:-rows_padding, cols_padding:-cols_padding]=img

  ero_img=np.zeros(shape=img.shape)

  for i in range(rows_padding, rows_padding+img.shape[0]):
    for j in range(cols_padding, cols_padding+img.shape[1]):
      row_range=slice(i-rows_padding, i+rows_padding+1)
      col_range=slice(j-cols_padding, j+cols_padding+1)

      ROI=img_padding[row_range, col_range]
      if(np.sum(ROI-SE)==0):
        ero_img[i, j]=1

  return ero_img

def dilation(img, SE=SE):
  # The padding of the rows and columns should depend on the SE.
  # Padding of 2 is used specifically when the SE is 3x3.
  rows_padding=SE.shape[0]//2
  cols_padding=SE.shape[1]//2
  img_padding=np.zeros(shape=(img.shape[0]+2*rows_padding, img.shape[1]+2*cols_padding))
  img_padding[rows_padding:-rows_padding, cols_padding:-cols_padding]=img

  dil_img=np.zeros(shape=img_padding.shape)

  for i in range(rows_padding, rows_padding+img.shape[0]):
    for j in range(cols_padding, cols_padding+img.shape[1]):
      row_range=slice(i-rows_padding, i+rows_padding+1)
      col_range=slice(j-cols_padding, j+cols_padding+1)

      if(img[i-rows_padding, j-cols_padding]):
        dil_img[row_range, col_range]=SE

  return dil_img