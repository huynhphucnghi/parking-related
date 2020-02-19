import numpy as np 
grid_size = 3

new = np.zeros((grid_size, grid_size, 5))
pos_y = 2
pos_x = 1
new[pos_y, pos_x, 0] = 1
new[pos_y, pos_x, 1] = 2
new[pos_y, pos_x, 2] = 3
new[pos_y, pos_x, 3] = 4
new[pos_y, pos_x, 4] = 1
print('origin')
print(new)
mask = np.expand_dims(new[..., 4], -1)
# print('mask')
# print(mask)
no_mask = 1 - mask
# print('no mask')
# print(no_mask)
new = no_mask * new
print('new')
print(new)
