# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 11:38:59 2023

@author: ashke
"""

import SimpleITK as sitk
import sys
import os
import matplotlib.pyplot as plt # for plotting the metric
from numpy import sign, zeros, max
from IPython.display import clear_output
import numpy as np
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import nibabel as nib
from nibabel.testing import data_path
import scipy as sp
import scipy.interpolate
from scipy import ndimage
import csv
from collections import defaultdict
import pandas as pd



def main(patient_id, img_number_1, img_number_2, img_filter):

    
    
    os.chdir("C:\Ashan\ct_data")
    referenceImage = sitk.ReadImage(patient_id+"_"+img_number_1+"_"+img_filter, sitk.sitkFloat32)
    os.chdir("C:\Ashan\masks")
    referenceMask = sitk.ReadImage(patient_id+"_"+img_number_1+"_"+img_filter, sitk.sitkFloat32)
    os.chdir("C:\Ashan\ct_data")
    floating = sitk.ReadImage(patient_id+"_"+img_number_2+"_"+img_filter, sitk.sitkFloat32)
    os.chdir("C:\Ashan\masks")
    floatingMask = sitk.ReadImage(patient_id+"_"+img_number_2+"_"+img_filter, sitk.sitkFloat32)
    
    
    # example_ni1 = os.path.join(data_path, '101089_0_STANDARD.nii')
    # n1_img = nib.load(example_ni1)
    # b = n1_img.header
    # print(b)
    
    # spacing = referenceImage.GetSpacing()
    # spacing_x = spacing[0]
    # spacing_y = spacing[1]
    # spacing_z = spacing[2]
    
    # determine volume of a single voxel
    # voxel_volume = spacing_x * spacing_y * spacing_z
    # print(spacing_x)
    # print(spacing_y)
    # print(spacing_z)
    # print(voxel_volume, "HERE")
    
    def start_plot():
        global metric_values, optimiser_iterations
        optimiser_iterations = []
        metric_values = []
        global fig, ax
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.ion()
        fig.show()
        fig.canvas.draw()
        
    def plot_values(registration_method):
        global metric_values, optimiser_iterations
        optimiser_iterations.append(registration_method.GetOptimizerIteration())
        metric_values.append(registration_method.GetMetricValue())                                       
        # Clear and plot the similarity metric values
    
        global fig, ax
    
        fig = plt.figure()
        ax = fig.add_subplot(111)
        clear_output()
            
        ax.clear()
        #ax.plot(optimiser_iterations, metric_values, 'b.')
        ax.plot( metric_values, 'b.')
        plt.xlabel('Iteration Number',fontsize=12)
        plt.ylabel('Metric Value',fontsize=12)
        ax.set_ylim([-1, 0])
        
        plt.show();
    
        fig.canvas.draw()
    def command_multires_iterations():
        print("    > ---- Resolution change ----")
        
    # Callback invoked by the IPython interact method for scrolling and modifying the alpha blending
    # of an image stack of two images that occupy the same physical space. 
    
    def resample_image_with_Tx(referenceImage, Tx, iimg):
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(referenceImage);
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetDefaultPixelValue(0)
        resampler.SetTransform(Tx)
        oimg = resampler.Execute(iimg)
        return oimg
    
    def display_image(Image1):
        sitk.Show(Image1, debugOn=True)
    
    def run_affine_registration(referenceImage, referenceMask, floatingImage, printInfo=True):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetInterpolator(sitk.sitkLinear)
        R.SetMetricFixedMask(referenceMask)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                    minStep=1e-2,
                                                    numberOfIterations=300,
                                                    gradientMagnitudeTolerance=1e-2,
                                                    maximumStepSizeInPhysicalUnits = 10)
        R.SetOptimizerScalesFromIndexShift()
        tx = sitk.CenteredTransformInitializer(referenceImage, floating, sitk.Similarity3DTransform(),
                                               sitk.CenteredTransformInitializerFilter.GEOMETRY)
        print("Initial Number of Parameters: {0}".format(tx.GetNumberOfParameters()))
        R.SetInitialTransform(tx)
        outTx = R.Execute(referenceImage, floatingImage)
        if printInfo:
            print("    >> Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
            print("    >> Iteration: {0}".format(R.GetOptimizerIteration()))
            print("    >> Metric value: {0}".format(R.GetMetricValue()))
        return outTx
    
    def run_rigid_registration(referenceImage, referenceMask, floatingImage, useMultiResolution=True, printInfo=True):
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetInterpolator(sitk.sitkLinear)
        R.SetMetricFixedMask(referenceMask)
        #R.SetMetricMovingMask(floatingMask)
        R.SetOptimizerAsRegularStepGradientDescent(learningRate=1.0,
                                                    minStep=1e-2,
                                                    numberOfIterations=300,
                                                    gradientMagnitudeTolerance=1e-2,
                                                    maximumStepSizeInPhysicalUnits = 10)
        
        
        
        R.SetOptimizerScalesFromIndexShift()
        tx = sitk.CenteredTransformInitializer(referenceImage, floating, sitk.Euler3DTransform())#,
                                               #sitk.CenteredTransformInitializerFilter.GEOMETRY)
        print("Initial Number of Parameters: {0}".format(tx.GetNumberOfParameters()))
        R.SetInitialTransform(tx)
        
        if useMultiResolution:
            print("here")
            R.SetShrinkFactorsPerLevel([8,4,2,1])
            R.SetSmoothingSigmasPerLevel([4,2,1,0])
            R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: command_multires_iterations() )
        
        
        outTx = R.Execute(referenceImage, floatingImage)
        if printInfo:
            print("    >> Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
            print("    >> Iteration: {0}".format(R.GetOptimizerIteration()))
            print("    >> Metric value: {0}".format(R.GetMetricValue()))
        return outTx
    
    
    
    rTx = run_rigid_registration(referenceImage, referenceMask, floating)
    floatingAfterRigid = resample_image_with_Tx(referenceImage, rTx, floating)
    
    def run_nonrigid_registration(referenceImage, referenceMask, floatingImage, useMultiResolution=True, printInfo=True):
        global R
        R = sitk.ImageRegistrationMethod()
        R.SetMetricAsCorrelation()
        R.SetInterpolator(sitk.sitkLinear)
        R.SetMetricFixedMask(referenceMask)
        R.SetOptimizerAsGradientDescentLineSearch(1.0, 100,
                                              convergenceMinimumValue=1e-4,
                                              convergenceWindowSize=5)
    
        transformDomainMeshSize=[8]*floatingImage.GetDimension()
        tx = sitk.BSplineTransformInitializer(referenceImage, transformDomainMeshSize )
        print("Initial Number of Parameters: {0}".format(tx.GetNumberOfParameters()))
        R.SetInitialTransform(tx, True)
        #R.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(R))
        if useMultiResolution:
            R.SetShrinkFactorsPerLevel([8,4,2,1])
            R.SetSmoothingSigmasPerLevel([4,2,1,0])
            R.AddCommand(sitk.sitkMultiResolutionIterationEvent, lambda: command_multires_iterations() )
        
        outTx = R.Execute(referenceImage, floatingImage)
        if printInfo:
            print("    >> Optimizer stop condition: {0}".format(R.GetOptimizerStopConditionDescription()))
            print("    >> Iteration: {0}".format(R.GetOptimizerIteration()))
            print("    >> Metric value: {0}".format(R.GetMetricValue()))
    
        displacement = sitk.TransformToDisplacementField(outTx, 
                                      sitk.sitkVectorFloat64,
                                      referenceImage.GetSize(),
                                      referenceImage.GetOrigin(),
                                      referenceImage.GetSpacing(),
                                      referenceImage.GetDirection())
    
    
        # Iterating over the slices for jacobian array
        global jacobian_det_np_arr
        global jacobian_det_volume
        jacobian_det_volume = sitk.DisplacementFieldJacobianDeterminant(displacement)
        jacobian_det_np_arr = sitk.GetArrayViewFromImage(jacobian_det_volume)
        return outTx
    
    
    #start_plot()
    nrrTx = run_nonrigid_registration(referenceImage, referenceMask, floatingAfterRigid)
    out = resample_image_with_Tx(referenceImage, nrrTx, floatingAfterRigid)
    #out_array = sitk.GetArrayFromImage(out)
    floatingMask_array = sitk.GetArrayFromImage(floatingMask)
    
    
    non_zero_slices = []
    
    for i in range(0,len(floatingMask_array)):
        slice_sum = sum(floatingMask_array[i].flatten())
        if slice_sum != 0:
            non_zero_slices.append(i)
            
    
    no_of_slices = np.size(non_zero_slices)
    begin_index = non_zero_slices[0]
    
    # Focusing on the tumour site using floatingMask 
    extract = sitk.ExtractImageFilter()
    extract.SetSize([60, 60, no_of_slices])
    extract.SetIndex([0, 0, begin_index])
    extracted_image = extract.Execute(out)
    cropped_jacobian_img = extract.Execute(jacobian_det_volume)
    cropped_jacobian = sitk.GetArrayViewFromImage(cropped_jacobian_img).transpose(2,1,0)
    cropped_floating_mask_img = extract.Execute(floatingMask)
    # Positive
    cropped_floating_mask = sitk.GetArrayViewFromImage(cropped_floating_mask_img).transpose(2,1,0)
    
    cropped_floating_mask_1 = np.copy(cropped_floating_mask)
    
    # Contouring the tumour site
    gtv = sitk.GetArrayViewFromImage(extracted_image).transpose(2,1,0)
    gtv1 = np.copy(gtv)
    
    
    # Contouring GTV
    for i in range(0,len(gtv1)):
        for j in range(0,len(gtv1[i,:,:])):
            for k in range(0,len(gtv1[i,j,:])):
                if cropped_floating_mask[i,j,k] == 0:
                    gtv1[i,j,k] = 0
                else:
                    gtv1[i,j,k] = 1
                    
    # Negative function on mask for distance transform 
    for i in range(0,len(cropped_floating_mask_1)):
        for j in range(0,len(cropped_floating_mask_1[i,:,:])):
            for k in range(0,len(cropped_floating_mask_1[i,j,:])):
                if cropped_floating_mask_1[i,j,k] == 0:
                    cropped_floating_mask_1[i,j,k] = 1
                else:
                    cropped_floating_mask_1[i,j,k] = 0
    # Plotting the GTV without background
    
    x_range = np.linspace(-1, 1, gtv1.shape[0])
    y_range = np.linspace(-1, 1, gtv1.shape[1])
    z_range = np.linspace(-1, 1, gtv1.shape[2])
    
    x, y, z = np.meshgrid(x_range, y_range,
                          z_range, indexing='ij')
    
    non_zero_mask = gtv1.flatten() != 0
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ax.scatter(x.flatten()[non_zero_mask], y.flatten()[non_zero_mask],
                z.flatten()[non_zero_mask], c=gtv1.flatten()[non_zero_mask],
                cmap='jet', marker='o', s=1)
    
    # Set labels for x, y, and z axes
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.axes.set_xlim3d(left=-1, right=1) 
    ax.axes.set_ylim3d(bottom=-1, top=1) 
    ax.axes.set_zlim3d(bottom=-1, top=1) 
    
    # Show the plot
    plt.show()
    
    
    def jacobian_at_margin(mask, boundary_size, title):
        
            # CREATE RANGE MASK IN 3D
            
        distance = scipy.ndimage.distance_transform_edt(mask, [0.5,0.5,0.5])
        x_range = np.linspace(-1, 1, distance.shape[0])
        y_range = np.linspace(-1, 1, distance.shape[1])
        z_range = np.linspace(-1, 1, distance.shape[2])
        
        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        non_zero_mask_ = distance.flatten() != 0
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(x.flatten()[non_zero_mask_], y.flatten()[non_zero_mask_],
                    z.flatten()[non_zero_mask_], c=distance.flatten()[non_zero_mask_],
                    cmap='jet', marker='o', s=1)
        
        
        # Set labels for x, y, and z axes
        
        ax.set_title('Slices of patient tumour site')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.axes.set_zlim3d(bottom=-1, top=1) 
        
        # # Show the plot
        # plt.show()
        
        pixels = []
    
        for depth, inner_arrays in enumerate(distance):
            for row, inner_array in enumerate(inner_arrays):
                for col, pixel in enumerate(inner_array):
                    if pixel > 0:
                        pixels.append(pixel)
        
        unique_pixels = list(set(pixels))
        pixel_location_dic = {}
        
        for pixel in unique_pixels:
            pixel_location_dic[pixel] = []
        
        for depth, inner_arrays in enumerate(distance):
            for row, inner_array in enumerate(inner_arrays):
                for col, pixel in enumerate(inner_array):
                    if pixel > 0:
                        pixel_location_dic[pixel].append([depth, row, col])
    
        keys = list(pixel_location_dic.keys())
    
        a = np.array(pixel_location_dic[keys[boundary_size]])
        
        jacob = np.zeros((60,60,no_of_slices))
        # for value in range(0,len(jacob)):
            
        
        jacobian_total = 0
        for l in a:
                jacob[l[0],l[1],l[2]] = cropped_jacobian[l[0],l[1],l[2]]
                jacobian_total+= cropped_jacobian[l[0],l[1],l[2]]
        
        
        mean_jacobian = jacobian_total/(np.size(a)/3)
    
        
        x, y, z = np.meshgrid(x_range, y_range, z_range, indexing='ij')
        
        non_zero_mask_ = jacob.flatten() != 0
    
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(x.flatten()[non_zero_mask_], y.flatten()[non_zero_mask_],
                    z.flatten()[non_zero_mask_], c=jacob.flatten()[non_zero_mask_],
                    cmap='jet', marker='o', s=1)
        
        
        # Set labels for x, y, and z axes
        ax.set_title(title)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_ylabel('Z')
        ax.axes.set_xlim3d(left=-1, right=1) 
        ax.axes.set_ylim3d(bottom=-1, top=1) 
        ax.axes.set_zlim3d(bottom=-1, top=1)
        cbar = plt.colorbar(ax.collections[0])
        cbar.set_label('Jacobian Determinant')
        
        
        # Show the plot
        plt.show()
        return mean_jacobian
    
    mean_jacobian_bound = jacobian_at_margin(cropped_floating_mask,0, "Jacobian determinant at boundary of GTV")
    mean_jacobian_outside = jacobian_at_margin(cropped_floating_mask_1,4,"Jacobian determinant 2mm outside the GTV")
    mean_jacobian_outside_ = jacobian_at_margin(cropped_floating_mask_1,8,"Jacobian determinant 4mm outside the GTV")
    mean_jacobian_outside__ = jacobian_at_margin(cropped_floating_mask_1,12,"Jacobian determinant 6mm outside the GTV")
    
    #mean_jacobian_outside_ = jacobian_at_margin(cropped_floating_mask,1,"Jacobian 0.5mm inside the GTV")
    #mean_jacobian_outside__ = jacobian_at_margin(cropped_floating_mask_1,2,"Jacobian 1mm inside the GTV")
    
    return mean_jacobian_bound, mean_jacobian_outside, mean_jacobian_outside_, mean_jacobian_outside__






main('101089', '0', '1', "STANDARD.nii")



# file_names = os.listdir("C:/Ashan/ct_data")

# count_dict = defaultdict(int)

# # Count occurrences of initial numbers and final words
# for file_name in file_names:
#     parts = file_name.split('_')
#     initial_number = parts[0]
#     final_word = parts[-1].split('.')[0]  # Remove '.nii' extension
#     count_dict[(initial_number, final_word)] += 1

# # Filter file names based on the conditions
# filtered_file_names = [
#     file_name for file_name in file_names
#     if count_dict[(file_name.split('_')[0], file_name.split('_')[-1].split('.')[0])] > 1
# ]


# jacobian_dictionary = {}
# images_registered = []



# # Loop for all images
# for file in filtered_file_names:
#     try:
#         split_name = file.split('_')
#         img_no = str(int(split_name[1]) + 1)
    
#         [m_jacobian,m_jacobian1,m_jacobian2, m_jacobian3] = main(split_name[0], split_name[1], img_no, split_name[2])
#         jacobian_list = []
#         jacobian_list.extend([m_jacobian, m_jacobian1, m_jacobian2, m_jacobian3])
#         jacobian_dictionary[split_name[0]+split_name[2]+img_no] = jacobian_list
#         images_registered.append(split_name[0])
#         print(file)

#     except:
#         print("Image number doesnt exist")
#         pass
    

# os.chdir("C:/Ashan")
# df = pd.DataFrame(jacobian_dictionary)
# new_df = df.transpose().reset_index()
# new_df.columns = ['key', '0', '1', '2', '3']
# excel_file_path = 'data.xlsx'
# new_df.to_excel(excel_file_path, index=False)





























