# -*- coding: utf-8 -*-
"""
Created on Sat Mar 11 15:11:49 2023

@author: Jyoti_Shukla
"""

from osgeo import gdal
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import MinMaxScaler


path = r"F:\Jyoti Shukla -MS\AVHRR Karnataka data (1981-2022)"
shapefile = r"F:\Jyoti Shukla -MS\Shapefiles for ROI\State.shp"
output_path = r"F:\Jyoti Shukla -MS\karnataka dataset\AVHRRR Karnataka dataset monthly"
dir_list = os.listdir(path)
array = np.zeros((190,124))
monthly = []
def monthly_club(target_list,year,i,k=0,j=0, array= np.zeros((190,124))):    
    for file in target_list:
        if file.endswith(".tif"):        
            image = os.path.join(path,file)
            ds = gdal.Open(image)
            array += ds.GetRasterBand(1).ReadAsArray()
            # print(i)
            i+=1
            if i%4 ==0 and i < 53:
                print("Merge finished for year:",year+j,"month:",k+1)
                array/=4
                monthly.append(array)
                array = np.zeros((190,124))
                k+=1
                # j+=1
            if i==52:
                k=0
                i=0
                j+=1
    # return monthly
#.....................for 1981-036 to 1981-052.............................................
target_list = dir_list[:18]
monthly_club(target_list,1981,35, k=8)


##............................for 1982-001 to 2021-052......................................
target_list = dir_list[18:2098]
monthly_club(target_list,1982,0)

###............................for 2022-001 to 2022-025......................................
target_list = dir_list[2098:]
monthly_club(target_list,2022,0)

np.save(output_path+"/Total_monthly_split_VHI_data_1981-2022.npy",monthly)

#......................................Scaling and splitting the data.......................
monthly_1981_2022 = np.load("F:\Jyoti Shukla -MS\karnataka dataset\AVHRRR Karnataka dataset monthly\Total_monthly_split_VHI_data_1981-2022.npy")
print(len(monthly_1981_2022))            

monthly_train = monthly_1981_2022[:481]
monthly_test = monthly_1981_2022[481:]
monthly_train_scaled =[]
monthly_test_scaled =[]
monthly_scaled = []
for i in range(len(monthly_train)):
    monthly_train[i] = np.where(monthly_train[i]<0,0,monthly_train[i]) #removing nodata
    # monthly_train[i+1] = np.where(monthly_train[i+1]<0,0,monthly_train[i+1])
    pad = np.pad(monthly_train[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
    # pad_label = np.pad(monthly_train[i+1],((2,0),(0,4)))
    #scaling in 0 to 1 range
    scaler = MinMaxScaler((0,100))
    scaled = scaler.fit_transform(pad)
    # scales.append(scaled)
    # scaled_label = scaler.fit_transform(pad_label)
    monthly_train_scaled.append((scaled.reshape(192,128,1))) # giving the next image as label for the current image
    # train_labels.append((scaled_label.reshape(192,128,1)))
print(monthly_train_scaled[1].shape)
plt.imshow(monthly_train_scaled[0].reshape(192,128))


for i in range(len(monthly_test)):
    monthly_test[i] = np.where(monthly_test[i]<0,0,monthly_test[i]) #removing nodata
    # monthly_train[i+1] = np.where(monthly_train[i+1]<0,0,monthly_train[i+1])
    pad = np.pad(monthly_test[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
    # pad_label = np.pad(monthly_train[i+1],((2,0),(0,4)))
    #scaling in 0 to 1 range
    scaler = MinMaxScaler((0,100))
    scaled = scaler.fit_transform(pad)
    # scales.append(scaled)
    # scaled_label = scaler.fit_transform(pad_label)
    monthly_test_scaled.append((scaled.reshape(192,128,1))) # giving the next image as label for the current image
    # train_labels.append((scaled_label.reshape(192,128,1)))
print(monthly_test_scaled[1].shape)
plt.imshow(monthly_test_scaled[0].reshape(192,128))


for i in range(len(monthly_1981_2022)):
    monthly_1981_2022[i] = np.where(monthly_1981_2022[i]<0,0,monthly_1981_2022[i]) #removing nodata
    # monthly_train[i+1] = np.where(monthly_train[i+1]<0,0,monthly_train[i+1])
    pad = np.pad(monthly_1981_2022[i],((2,0),(0,4))) # padding to get divisible by 2 shape (190,128)
    # pad_label = np.pad(monthly_train[i+1],((2,0),(0,4)))
    #scaling in 0 to 1 range
    scaler = MinMaxScaler((0,100))
    scaled = scaler.fit_transform(pad)
    # scales.append(scaled)
    # scaled_label = scaler.fit_transform(pad_label)
    monthly_scaled.append((scaled.reshape(192,128,1))) # giving the next image as label for the current image
    # train_labels.append((scaled_label.reshape(192,128,1)))
print(monthly_scaled[1].shape)
plt.imshow(monthly_scaled[0].reshape(192,128))


np.save(output_path+"/monthly_train_scaled100_avhrr_vhi_1981_2022.npy",monthly_train_scaled)
np.save(output_path+"/monthly_test_scaled100_avhrr_vhi_1981_2022.npy",monthly_test_scaled)
np.save(output_path+"/monthly_scaled100_avhrr_vhi_1981_2022.npy",monthly_scaled)


##.......................Temporal labeling the scaled data.....................................
monthly_scaled = np.load("F:\Jyoti Shukla -MS\karnataka dataset\AVHRRR Karnataka dataset monthly\monthly_scaled100_avhrr_vhi_1981_2022.npy")


def temporal_label(time, month):
    data =[]
    label =[]
    data_norm =[]
    label_norm =[]
    for i in range(len(month)-time):
        data.append(month[i])
        label.append(month[i+time])
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(month[i].reshape(192,128))
        # scales.append(scaled)
        scaled_label = scaler.fit_transform(month[i+time].reshape(192,128))
        data_norm.append((scaled.reshape(192,128,1))) # giving the next image as label for the current image
        label_norm.append((scaled_label.reshape(192,128,1)))
    return data, label,data_norm, label_norm
        
#.........................an year gap.......................................................
data_12,label_12, data_norm_12, label_norm_12 = temporal_label(12, monthly_scaled)

np.save(output_path+"/12monthly_scaled100_avhrr_vhi_1981_2022.npy",data_12)
np.save(output_path+"/12monthly_scaled100_avhrr_vhi_1981_2022_label.npy",label_12)
np.save(output_path+"/12monthly_scaledto1_avhrr_vhi_1981_2022.npy",data_norm_12)
np.save(output_path+"/12monthly_scaledto1_avhrr_vhi_1981_2022_label.npy",label_norm_12)
train_12, train_label_12 = data_12[:474], label_12[:474]
test_12, test_label_12 = data_12[474:], label_12[474:]
train_norm_12, train_label_norm_12 = data_norm_12[:474], label_norm_12[:474]
test_norm_12, test_label_norm_12 = data_norm_12[474:], label_norm_12[474:]
# normalized train pairing
x = train_norm_12
y = train_label_norm_12
np.savez(output_path+"/12monthly_train_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real train pairing
x = train_12
y = train_label_12
np.savez(output_path+"/12monthly_train_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized test pairing
x = test_norm_12
y = test_label_norm_12
np.savez(output_path+"/12monthly_test_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real test pairing
x = test_12
y = test_label_12
np.savez(output_path+"/12monthly_test_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized total pairing
x = data_norm_12
y = label_norm_12
np.savez(output_path+"/12monthly_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)

#........................9-monthly..........................................................
data_09,label_09, data_norm_09, label_norm_09 = temporal_label(9, monthly_scaled)

np.save(output_path+"/9monthly_scaled100_avhrr_vhi_1981_2022.npy",data_09)
np.save(output_path+"/9monthly_scaled100_avhrr_vhi_1981_2022_label.npy",label_09)
np.save(output_path+"/9monthly_scaledto1_avhrr_vhi_1981_2022.npy",data_norm_09)
np.save(output_path+"/9monthly_scaledto1_avhrr_vhi_1981_2022_label.npy",label_norm_09)
train_09, train_label_09 = data_09[:477], label_09[:477]
test_09, test_label_09 = data_09[477:], label_09[477:]
train_norm_09, train_label_norm_09 = data_norm_09[:477], label_norm_09[:477]
test_norm_09, test_label_norm_09 = data_norm_09[477:], label_norm_09[477:]
# normalized train pairing
x = train_norm_09
y = train_label_norm_09
np.savez(output_path+"/9monthly_train_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real train pairing
x = train_09
y = train_label_09
np.savez(output_path+"/9monthly_train_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized test pairing
x = test_norm_09
y = test_label_norm_09
np.savez(output_path+"/9monthly_test_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real test pairing
x = test_09
y = test_label_09
np.savez(output_path+"/9monthly_test_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized total pairing
x = data_norm_09
y = label_norm_09
np.savez(output_path+"/9monthly_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)

#........................6-monthly..........................................................
data_06,label_06, data_norm_06, label_norm_06 = temporal_label(6, monthly_scaled)

np.save(output_path+"/6monthly_scaled100_avhrr_vhi_1981_2022.npy",data_06)
np.save(output_path+"/6monthly_scaled100_avhrr_vhi_1981_2022_label.npy",label_06)
np.save(output_path+"/6monthly_scaledto1_avhrr_vhi_1981_2022.npy",data_norm_06)
np.save(output_path+"/6monthly_scaledto1_avhrr_vhi_1981_2022_label.npy",label_norm_06)
train_06, train_label_06 = data_06[:480], label_06[:480]
test_06, test_label_06 = data_06[480:], label_06[480:]
train_norm_06, train_label_norm_06 = data_norm_06[:480], label_norm_06[:480]
test_norm_06, test_label_norm_06 = data_norm_06[480:], label_norm_06[480:]
# normalized train pairing
x = train_norm_06
y = train_label_norm_06
np.savez(output_path+"/6monthly_train_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real train pairing
x = train_06
y = train_label_06
np.savez(output_path+"/6monthly_train_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized test pairing
x = test_norm_06
y = test_label_norm_06
np.savez(output_path+"/6monthly_test_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real test pairing
x = test_06
y = test_label_06
np.savez(output_path+"/6monthly_test_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized total pairing
x = data_norm_06
y = label_norm_06
np.savez(output_path+"/6monthly_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
#........................3-monthly..........................................................
data_03,label_03, data_norm_03, label_norm_03 = temporal_label(3, monthly_scaled)

np.save(output_path+"/3monthly_scaled100_avhrr_vhi_1981_2022.npy",data_03)
np.save(output_path+"/3monthly_scaled100_avhrr_vhi_1981_2022_label.npy",label_03)
np.save(output_path+"/3monthly_scaledto1_avhrr_vhi_1981_2022.npy",data_norm_03)
np.save(output_path+"/3monthly_scaledto1_avhrr_vhi_1981_2022_label.npy",label_norm_03)
train_03, train_label_03 = data_03[:483], label_03[:483]
test_03, test_label_03 = data_03[483:], label_03[483:]
train_norm_03, train_label_norm_03 = data_norm_03[:483], label_norm_03[:483]
test_norm_03, test_label_norm_03 = data_norm_03[483:], label_norm_03[483:]
# normalized train pairing
x = train_norm_03
y = train_label_norm_03
np.savez(output_path+"/3monthly_train_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real train pairing
x = train_03
y = train_label_03
np.savez(output_path+"/3monthly_train_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized test pairing
x = test_norm_03
y = test_label_norm_03
np.savez(output_path+"/3monthly_test_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real test pairing
x = test_03
y = test_label_03
np.savez(output_path+"/3monthly_test_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized total pairing
x = data_norm_03
y = label_norm_03
np.savez(output_path+"/3monthly_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)

#........................1-monthly..........................................................
data_01,label_01, data_norm_01, label_norm_01 = temporal_label(1, monthly_scaled)

np.save(output_path+"/1monthly_scaled100_avhrr_vhi_1981_2022.npy",data_01)
np.save(output_path+"/1monthly_scaled100_avhrr_vhi_1981_2022_label.npy",label_01)
np.save(output_path+"/1monthly_scaledto1_avhrr_vhi_1981_2022.npy",data_norm_01)
np.save(output_path+"/1monthly_scaledto1_avhrr_vhi_1981_2022_label.npy",label_norm_01)
train_01, train_label_01 = data_01[:485], label_01[:485]
test_01, test_label_01 = data_01[485:], label_01[485:]
train_norm_01, train_label_norm_01 = data_norm_01[:485], label_norm_01[:485]
test_norm_01, test_label_norm_01 = data_norm_01[485:], label_norm_01[485:]
# normalized train pairing
x = train_norm_01
y = train_label_norm_01
np.savez(output_path+"/1monthly_train_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real train pairing
x = train_01
y = train_label_01
np.savez(output_path+"/1monthly_train_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized test pairing
x = test_norm_01
y = test_label_norm_01
np.savez(output_path+"/1monthly_test_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)
# real test pairing
x = test_01
y = test_label_01
np.savez(output_path+"/1monthly_test_paired_scaled100_avhrr_vhi_1981_2022.npz",x = x , y = y)
# normalized total pairing
x = data_norm_01
y = label_norm_01
np.savez(output_path+"/1monthly_paired_scaledto1_avhrr_vhi_1981_2022.npz",x = x , y = y)