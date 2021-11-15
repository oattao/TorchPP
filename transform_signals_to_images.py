import os 
import shutil
import glob
import cv2 as cv
from utils.signal import read_mat_file, add_noise, signal_to_gray_image, signal_to_scalorgram_image


data_path = os.path.join('.', 'data', 'original_signals')
output_path = os.path.join('.', 'data', 'transformed_images')
# noise_levels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# horse_powers = [0, 1, 2, 3]
noise_levels = ['nonoise']
horse_powers = [0]
fault_types = ['NoFault', 
               'Inner7', 'Inner14', 'Inner21',
               'Ball7', 'Ball14', 'Ball21', 
               'Outer7', 'Outer14', 'Outer21']
horse_power_0_files = [97, 105, 118, 130, 169, 185, 197, 209, 222, 234] 

LENGTH = 784
OVERLAP = int(0.5 * LENGTH)

for hp in horse_powers:
    hp_output = os.path.join(output_path, f'hp_{hp}')
    if os.path.exists(hp_output):
        shutil.rmtree(hp_output)
    os.mkdir(hp_output)

    listfiles = [fname + hp for fname in horse_power_0_files]

    for nlv in noise_levels:
        noise_output = os.path.join(hp_output, f'noise_{nlv}_db')
        os.mkdir(noise_output)

        for i, fault_name in enumerate(fault_types):
            print(f'Processing HP: {hp}, Noise level: {nlv}, File: {listfiles[i]}')

            fault_output = os.path.join(noise_output, fault_name)
            gray_output = os.path.join(fault_output, 'gray')
            scalo_output = os.path.join(fault_output, 'scalo')
            os.mkdir(fault_output)
            os.mkdir(gray_output)
            os.mkdir(scalo_output)

            file_name = listfiles[i]
            mat_path = os.path.join(data_path, f'{file_name}.mat')
            signal = read_mat_file(mat_path)
            if nlv != 'nonoise':
                signal = add_noise(signal, nlv)

            start = 0
            cnt = 0
            while cnt < 300:
                end = start + LENGTH
                sig = signal[start: end]
                if len(sig) < LENGTH:
                    break

                gray_img = signal_to_gray_image(sig)
                scalo_img = signal_to_scalorgram_image(sig)
                cv.imwrite(os.path.join(gray_output, f'{cnt}.png'), gray_img)
                cv.imwrite(os.path.join(scalo_output, f'{cnt}.png'), scalo_img)
                start = end - OVERLAP
                cnt += 1