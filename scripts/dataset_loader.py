import os
from torch.utils.data import Dataset
from PIL import Image
import random
import json
import re


def numerical_sort_key(filename):
    # Find all numbers in the filename
    numbers = re.findall(r"\d+", filename)
    # Return the last number as an integer (or 0 if no numbers found)
    return int(numbers[-1]) if numbers else 0


class TrainingDataset(Dataset):
    def __init__(self, dataset_paths, mode="image", transform=None, ratio=None):
        """
        Args:
            dataset_paths (list): List of paths to the dataset directories.
            mode (str): Either 'image' (image-level training) or 'patient' (patient-level training). Both data fields are populated regardless of mode. 'image' by default
            transform (callable, optional): Transformations to apply to the images.
            ratio (number, optional): Segmentation ratio (ratio between images with tumor present and images with tumor not present)
        """
        if mode not in ["image", "patient"]:
            raise ValueError("Mode must be either 'image' or 'patient'")

        self.transform = transform
        self.mode = mode

        self.data = (
            []
        )  # To store (image_path, mask_path, patient_id, scan_type, segmentation_flag) tuples
        # segmentation_flag refers to whether there is tumor in the mask or not (only zeros or not)
        self.filtered_data = []  # To store filtered data based on segmentation ratio
        self.patient_data = {}  # To store patient_id as key, patient images as values
        self.filtered_patient_data = (
            {}
        )  # To store filtered patient data based on segmentation ratio

        self.mask_segmentation_path = os.getcwd() + "/" + "mask_segmentations.json"

        mask_segmentations = None
        with open(self.mask_segmentation_path, "r") as f:
            mask_segmentations = json.load(f)

        for dataset_path in dataset_paths:
            image_number = 0
            patient_number = 0
            patients = os.listdir(dataset_path)

            for patient in patients:
                patient_path = os.path.join(dataset_path, patient)

                if not os.path.isdir(patient_path):
                    continue

                patient_number += 1

                masks_path = None
                images_paths = []

                # Locate segmentation (mask) and image directories
                for folder in os.listdir(patient_path):
                    folder_path = os.path.join(patient_path, folder)
                    if "segmentation" in folder.lower():
                        masks_path = folder_path
                    else:
                        images_paths.append(folder_path)

                # Verify directories exist
                if len(images_paths) == 0 or not os.path.isdir(masks_path):
                    print(f"Skipping patient {patient}: missing images or masks")
                    continue

                # Process images and masks for each modality
                for images_path in images_paths:
                    scan_type = os.path.basename(images_path)  # "FLAIR" or "T1" or "T2"
                    scan_type = scan_type.upper()
                    image_files = os.listdir(images_path)
                    mask_files = os.listdir(masks_path)
                    # Mask dictionary -  key slice number, value mask path
                    mask_dict = {}
                    for mask_file in mask_files:
                        slice_number = int(re.findall(r"\d+", mask_file)[-1])
                        mask_dict[slice_number] = mask_file

                    image_number += len(image_files)

                    # Match images and masks by slice number
                    for img_file in image_files:
                        # Slice number is the last number in the filename
                        slice_number = int(re.findall(r"\d+", img_file)[-1])
                        # Find the corresponding mask file
                        mask_file = mask_dict.get(slice_number)
                        img_path = os.path.join(images_path, img_file)
                        mask_path = os.path.join(masks_path, mask_file)

                        key = patient + "-" + mask_file

                        segmentation_flag = mask_segmentations[key]

                        # Add image-mask pair for image-level training
                        self.data.append(
                            (img_path, mask_path, patient, scan_type, segmentation_flag)
                        )
                        # Organize by patient for patient-level training
                        self.patient_data.setdefault(patient, []).append(
                            (img_path, mask_path, scan_type, segmentation_flag)
                        )

            print(
                f"Dataset {dataset_path} loaded with {image_number} images, and {patient_number} patients"
            )

        # Print summary
        print(f"Loaded {len(self.data)} image-mask pairs for image-level training")
        print(f"Loaded {len(self.patient_data)} patients for patient-level training")

        segmented_data = [data for data in self.data if data[4] == 1]
        non_segmented_data = [data for data in self.data if data[4] == 0]

        print(
            f"Found {len(segmented_data)} image-mask pairs with tumor present and {len(non_segmented_data)} image-mask pair with tumor not present. Ratio {len(segmented_data) / len(self.data)}"
        )

        self.set_segmentation_ratio(ratio)

    def reset(self):
        """
        Reset the dataset to its original state.
        """
        self.filtered_data = self.data
        self.filtered_patient_data = self.patient_data
        self.ratio = None

    def __len__(self):
        """
        Returns the length of the dataset.
        """
        if self.mode == "image":
            return len(self.filtered_data)
        elif self.mode == "patient":
            return len(self.filtered_patient_data)

    def __getitem__(self, idx):
        """
        Retrieves a single data point based on the mode.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (image, mask, metadata) where metadata includes patient_id and scan_type.
        """
        if self.mode == "image":
            # Image-level training: return individual image-mask pair
            img_path, mask_path, patient_id, scan_type, segmentation_flag = (
                self.filtered_data[idx]
            )
        elif self.mode == "patient":
            # Patient-level training: select one random image-mask pair for a given patient
            patient_id = list(self.filtered_patient_data.keys())[idx]
            patient_images = self.filtered_patient_data[patient_id]

            # Randomly select one image-mask pair for this patient
            img_path, mask_path, scan_type, segmentation_flag = random.choice(
                patient_images
            )

        # Load image and mask
        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")  # Assuming masks are grayscale

        # Apply transforms if available
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Binarize tensor mask
        mask = (mask > 0).float()

        return (
            image,
            mask,
            {
                "patient_id": patient_id,
                "scan_type": scan_type,
                "segmentation_flag": segmentation_flag,
            },
        )

    def change_mode(self, mode):
        """
        Change the mode of the dataset between 'image' and 'patient'.
        Args:
            mode (str): Either 'image' or 'patient'.
        """
        if mode not in ["image", "patient"]:
            raise ValueError("Mode must be either 'image' or 'patient'")

        self.mode = mode
        self.set_segmentation_ratio(None)

        if self.mode == "image":
            print(
                f"Switched to 'image' mode: {len(self.filtered_data)} image-mask pairs available. Ratio: {self.ratio}"
            )
        elif self.mode == "patient":
            print(
                f"Switched to 'patient' mode: Data available for {len(self.filtered_patient_data)} patients. Ratio: {self.ratio}"
            )

    def set_segmentation_ratio(self, ratio):
        """
        Set the segmentation ratio for the dataset in percentage and filter the data accordingly.
        Args:
            ratio (number): Segmentation ratio (ratio between images containing segmentation and images without segmentation)
        """

        self.reset()

        if ratio == None:
            self.ratio = ratio
            return

        if ratio < 0 or ratio > 1:
            raise ValueError("Segmentation ratio must be between 0 and 1")

        self.ratio = ratio

        if self.mode == "image":
            segmented_data = [data for data in self.data if data[4] == 1]
            non_segmented_data = [data for data in self.data if data[4] == 0]

            num_segmented = int(len(self.data) * ratio)
            num_non_segmented = int(len(self.data) * (1 - ratio))

            if num_segmented > len(segmented_data):
                # This is the case where we need to take all the segmented data and some non-segmented data in order to reach the desired ratio
                # Number of segmented images is fixed by the dataset
                num_segmented = len(segmented_data)

                # Calculate the required number of non-segmented images
                num_non_segmented = int((1 - ratio) / ratio * num_segmented)

                # Handle case where there aren't enough non-segmented images
                if num_non_segmented > len(non_segmented_data):
                    num_non_segmented = len(non_segmented_data)

                # Create the filtered dataset
                filtered_non_segmented = random.sample(
                    non_segmented_data, num_non_segmented
                )
                self.filtered_data = segmented_data + filtered_non_segmented
                random.shuffle(self.filtered_data)

                print(
                    f"Filtered dataset contains {len(segmented_data)} image-mask pairs with tumor present and {len(filtered_non_segmented)} image-mask pairs with no tumor present, ratio: {len(segmented_data) / len(self.filtered_data)}. Total: {len(self.filtered_data)}"
                )
            else:
                # This is the case where we DO NOT need to take all the non-segmented data in order to reach the desired ratio
                # Check if the desired number of non-segmented images is available
                if num_non_segmented > len(non_segmented_data):
                    # Not enough non-segmented images, adjust the ratio by reducing segmented images
                    num_non_segmented = len(non_segmented_data)
                    num_segmented = int(num_non_segmented * (ratio / (1 - ratio)))

                # Create the filtered dataset
                filtered_segmented = random.sample(segmented_data, num_segmented)
                filtered_non_segmented = random.sample(
                    non_segmented_data, num_non_segmented
                )
                self.filtered_data = filtered_segmented + filtered_non_segmented
                random.shuffle(self.filtered_data)

                print(
                    f"Filtered dataset contains {len(filtered_segmented)} image-mask pairs with tumor present and {len(filtered_non_segmented)} image-mask with tumor not present pairs, ratio: {len(filtered_segmented) / len(self.filtered_data)} Total: {len(self.filtered_data)}"
                )

        elif self.mode == "patient":
            # In this case ratio refers to the number of patients with tumor vs the number of patients without tumor (ONE IMAGE PER PATIENT)
            self.filtered_patient_data = {}
            patients = list(self.patient_data.keys())
            num_patients = len(patients)

            # Determine the number of patients with tumor-present images
            num_tumor = int(num_patients * ratio)
            num_non_tumor = num_patients - num_tumor

            tumor_patients = []
            non_tumor_patients = []

            # Separate patients based on whether they have tumor-present images
            for patient_id in patients:
                images = self.patient_data[patient_id]
                tumor_images = [img for img in images if img[3] == 1]
                non_tumor_images = [img for img in images if img[3] == 0]

                if tumor_images:
                    tumor_patients.append((patient_id, tumor_images))
                if non_tumor_images:
                    non_tumor_patients.append((patient_id, non_tumor_images))

            # Select patients to ensure the desired ratio
            selected_tumor_patients = random.sample(
                tumor_patients, min(num_tumor, len(tumor_patients))
            )
            selected_tumor_patient_ids = [
                patient[0] for patient in selected_tumor_patients
            ]
            remaining_non_tumor_patients = [
                patient
                for patient in non_tumor_patients
                if patient[0] not in selected_tumor_patient_ids
            ]
            selected_non_tumor_patients = random.sample(
                remaining_non_tumor_patients,
                min(num_non_tumor, len(remaining_non_tumor_patients)),
            )

            # For each selected patient, choose one image (tumor or non-tumor)
            for patient_id, images in selected_tumor_patients:
                self.filtered_patient_data[patient_id] = [
                    random.choice(images)
                ]  # Pick one tumor image

            for patient_id, images in selected_non_tumor_patients:
                non_tumor_images = [img for img in images if img[3] == 0]
                self.filtered_patient_data[patient_id] = [
                    random.choice(non_tumor_images)
                ]  # Pick one non-tumor image

            print(
                f"Filtered data contains {len(selected_tumor_patients)} patients with tumor present and {len(selected_non_tumor_patients)} patients with no tumor present, ratio: {len(selected_tumor_patients) / len(self.filtered_patient_data)}"
            )


class ValidationDataset(Dataset):
    def __init__(self, validation_dataset_paths, transform=None):
        self.data = (
            []
        )  # To store (image_path, mask_path, patient_id, segmentation_flag) tuples
        # segmentation_flag refers to whether there is tumor in the mask or not (only zeros or not)
        self.transform = transform
        self.mask_segmentation_path = os.getcwd() + "/" + "mask_segmentations.json"
        mask_segmentations = None
        with open(self.mask_segmentation_path, "r") as f:
            mask_segmentations = json.load(f)

        for dataset_path in validation_dataset_paths:
            image_number = 0

            patients = os.listdir(dataset_path)

            for patient in patients:
                patient_path = os.path.join(dataset_path, patient)

                if not os.path.isdir(patient_path):
                    continue

                masks_path = None
                images_paths = []

                # Locate segmentation (mask) and image directories
                for folder in os.listdir(patient_path):
                    folder_path = os.path.join(patient_path, folder)
                    if "segmentation" in folder.lower():
                        masks_path = folder_path
                    else:
                        images_paths.append(folder_path)

                # Verify directories exist
                if len(images_paths) == 0 or not os.path.isdir(masks_path):
                    print(f"Skipping patient {patient}: missing images or masks")
                    continue

                for images_path in images_paths:
                    # Process images and masks
                    scan_type = os.path.basename(images_path)  # "FLAIR" or "T1" or "T2"
                    scan_type = scan_type.upper()
                    image_files = os.listdir(images_path)
                    mask_files = os.listdir(masks_path)
                    # Mask dictionary -  key slice number, value mask path
                    mask_dict = {}
                    for mask_file in mask_files:
                        slice_number = int(re.findall(r"\d+", mask_file)[-1])
                        mask_dict[slice_number] = mask_file

                    image_number += len(image_files)

                    # Match images and masks by slice number
                    for img_file in image_files:
                        # Slice number is the last number in the filename
                        slice_number = int(re.findall(r"\d+", img_file)[-1])
                        # Find the corresponding mask file
                        mask_file = mask_dict.get(slice_number)
                        img_path = os.path.join(images_path, img_file)
                        mask_path = os.path.join(masks_path, mask_file)

                        key = patient + "-" + mask_file

                        segmentation_flag = mask_segmentations[key]

                        # Add image-mask pair for image-level training
                        self.data.append(
                            (img_path, mask_path, patient, scan_type, segmentation_flag)
                        )

            print(f"Dataset {dataset_path} loaded with {image_number} images.")

        print(f"Loaded {len(self.data)} image-mask pairs for validation")

        segmented_data = [data for data in self.data if data[4] == 1]
        non_segmented_data = [data for data in self.data if data[4] == 0]

        print(
            f"Found {len(segmented_data)} image-mask pairs with tumor present and {len(non_segmented_data)} image-mask pair with tumor not present. Ratio {len(segmented_data) / len(self.data)}"
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, mask_path, patient_id, scan_type, segmentation_flag = self.data[idx]

        image = Image.open(img_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        # Binarize tensor mask
        mask = (mask > 0).float()

        return (
            image,
            mask,
            {
                "patient_id": patient_id,
                "scan_type": scan_type,
                "segmentation_flag": segmentation_flag,
            },
        )
