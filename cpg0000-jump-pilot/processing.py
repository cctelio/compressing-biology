import os
import argparse
import pandas as pd
import re
import numpy as np
from PIL import Image

def extract_info_from_filename(filename):
    pattern = r'r(\d+)c(\d+)f(\d+)p\d+-ch(\d+)'
    match = re.match(pattern, filename)
    if match:
        row = int(match.group(1))
        column = int(match.group(2))
        fov = int(match.group(3))
        channel = int(match.group(4))
        return row, column, fov, channel
    return None

def create_dataframe_from_images(folder_path, correction_arrays, output_folder, assay_plate_barcode):
    os.makedirs(output_folder, exist_ok=True)
    data = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.tiff'):
            info = extract_info_from_filename(filename)
            if info:
                row, column, fov, channel = info
                image_path = os.path.join(folder_path, filename)

                # Load and correct image
                image = Image.open(image_path)
                image_np = np.array(image)
                correction_array = correction_arrays[channel]
                correction_array[correction_array == 0] = 1e-6 #Avoiding dividing by 0
                corrected_image_np = image_np / correction_array
                corrected_image_np = np.clip(corrected_image_np / 256, 0, 255).astype(np.uint8)

                # Save corrected image
                corrected_image = Image.fromarray(corrected_image_np)
                corrected_image_path = os.path.join(output_folder, filename.replace(".tiff", ".png"))
                corrected_image.save(corrected_image_path, format="PNG")

                data.append({
                    'row': row,
                    'column': column,
                    'FOV': fov,
                    'channel': channel,
                    'path': corrected_image_path,
                })

    df = pd.DataFrame(data)
    df_pivot = df.pivot_table(index=['row', 'column', 'FOV'], columns='channel', values='path', aggfunc='first').reset_index()
    df_pivot.columns = [f'channel_{col}' if isinstance(col, int) else col for col in df_pivot.columns]
    df_pivot['Assay_Plate_Barcode'] = assay_plate_barcode

    return df_pivot

def main_worker(args):

    path_images_temp = os.path.join(args.path_dataset, "images", args.assay_plate_barcode)
    [folder_images] = [f for f in os.listdir(path_images_temp) if os.path.isdir(os.path.join(path_images_temp, f))]
    path_images = os.path.join(path_images_temp, folder_images, "Images")

    path_illum = os.path.join(args.path_dataset, "illum", args.assay_plate_barcode)
    correction_arrays = {
        1: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumMito.npy")),        # Alexa 647
        2: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumAGP.npy")),         # Alexa 568
        3: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumRNA.npy")),         # Alexa 488 long
        4: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumER.npy")),          # Alexa 488
        5: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumDNA.npy")),         # Hoechst 33342
        6: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumLowZBF.npy")),      # Brightfield Z-plane (low)
        7: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumBrightfield.npy")), # Brightfield Z-plane (middle)
        8: np.load(os.path.join(path_illum, args.assay_plate_barcode + "_IllumHighZBF.npy"))      # Brightfield Z-plane (high)
    }

    processed_folder = os.path.join(args.path_dataset, "processed")
    os.makedirs(processed_folder, exist_ok=True)
    output_folder = os.path.join(processed_folder, args.assay_plate_barcode)

    df = create_dataframe_from_images(path_images, correction_arrays, output_folder, args.assay_plate_barcode)

    path_csv = os.path.join(args.path_dataset, "csv")
    os.makedirs(path_csv, exist_ok=True)
    df.to_csv(os.path.join(path_csv, args.assay_plate_barcode + ".csv"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--assay_plate_barcode', default='BR00116991', type=str)
    parser.add_argument('--path_dataset', default='/folder1/folder2/cpg0000-jump-pilot', type=str)

    args = parser.parse_args()
    
    # Spawn processes
    main_worker(args=args)

if __name__=="__main__":
    main()