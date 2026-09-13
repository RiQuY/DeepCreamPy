import numpy as np
from PIL import Image, ImageFilter
import os

from copy import deepcopy

import config
from libs.pconv_hybrid_model import PConvUnet
from libs.flood_fill import find_regions, expand_bounding

class Decensor():

    def __init__(self):
        self.args = config.get_args()
        self.is_mosaic = self.args.is_mosaic

        self.mask_color = [self.args.mask_color_red/255.0, self.args.mask_color_green/255.0, self.args.mask_color_blue/255.0]

        if not os.path.exists(self.args.decensor_output_path):
            os.makedirs(self.args.decensor_output_path)

        self.load_model()

    def get_mask(self, colored):
        colored_array = np.asarray(colored.convert('RGB'))

        mask = np.ones(
            (1, colored_array.shape[0], colored_array.shape[1], 3),
            dtype=np.uint8
        )

        green_pixels = np.all(
            colored_array == np.array([
                self.args.mask_color_red,
                self.args.mask_color_green,
                self.args.mask_color_blue
            ], dtype=np.uint8),
            axis=-1
        )

        # print("EXACT [0,255,0] PIXELS:", np.sum(green_pixels))

        mask[0, green_pixels] = 0

        # print(
        #     "masked pixels:",
        #     np.sum(mask == 0),
        #     "of",
        #     mask.size,
        #     "mask min/max:",
        #     mask.min(),
        #     mask.max()
        # )

        return mask

    def load_model(self):
        self.model = PConvUnet(weight_filepath='data/logs/')
        self.model.load(
            r"./models/model.h5",
            train_bn=False,
            lr=0.00005
        )

    def decensor_all_images_in_folder(self):
        #load model once at beginning and reuse same model
        #self.load_model()
        color_dir = self.args.decensor_input_path
        file_names = os.listdir(color_dir)

        #convert all images into np arrays and put them in a list
        for file_name in file_names:
            color_file_path = os.path.join(color_dir, file_name)
            if os.path.isfile(color_file_path) and os.path.splitext(color_file_path)[1] == ".png":
                print("--------------------------------------------------------------------------")
                print("Decensoring the image {color_file_path}".format(color_file_path = color_file_path))
                colored_img = Image.open(color_file_path)
                #if we are doing a mosaic decensor
                if self.is_mosaic:
                    #get the original file that hasn't been colored
                    ori_dir = self.args.decensor_input_original_path
                    #since the original image might not be a png, test multiple file formats
                    valid_formats = {".png", ".jpg", ".jpeg"}
                    found_valid = False
                    for valid_format in valid_formats:
                        test_file_name = os.path.splitext(file_name)[0] + valid_format
                        ori_file_path = os.path.join(ori_dir, test_file_name)
                        if os.path.isfile(ori_file_path):
                            found_valid = True
                            ori_img = Image.open(ori_file_path)
                            self.decensor_image(ori_img, colored_img, file_name)
                            continue
                    if not found_valid:
                        print("Corresponding original, uncolored image not found in {ori_file_path}. \nCheck if it exists and is in the PNG or JPG format.".format(ori_file_path = ori_file_path))
                else:
                    self.decensor_image(colored_img, colored_img, file_name)
        print("--------------------------------------------------------------------------")

    #decensors one image at a time
    #TODO: decensor all cropped parts of the same image in a batch (then i need input for colored an array of those images and make additional changes)
    def decensor_image(self, ori, colored, file_name):
        width, height = ori.size
        #save the alpha channel if the image has an alpha channel
        has_alpha = False
        if (ori.mode == "RGBA"):
            has_alpha = True
            alpha_channel = np.asarray(ori)[:,:,3]
            alpha_channel = np.expand_dims(alpha_channel, axis =-1)
            ori = ori.convert('RGB')

        ori_array = np.asarray(ori)
        ori_array = np.array(ori_array / 255.0)
        ori_array = np.expand_dims(ori_array, axis = 0)

        if self.is_mosaic:
            mask = np.ones(ori_array.shape, dtype=np.uint8)
        else:
            mask = self.get_mask(colored)


        #colored image is only used for finding the regions
        regions = find_regions(colored.convert('RGB'))
        print("Found {region_count} censored regions in this image!".format(region_count = len(regions)))
        
        # ------------------------------------------------------------------
        # DEBUG: verify that EVERY exact-green pixel belongs to a processed
        # region bounding box.
        # ------------------------------------------------------------------

        green_mask_2d = np.all(
            np.asarray(colored.convert('RGB')) == [0, 255, 0],
            axis=2
        )

        green_y, green_x = np.where(green_mask_2d)

        # print("GLOBAL EXACT GREEN PIXELS:", len(green_x))

        covered = np.zeros_like(green_mask_2d, dtype=bool)

        for i, region in enumerate(regions):

            # region is a collection of (x, y) pixels.
            region_pixels = np.asarray(list(region))

            if region_pixels.size == 0:
                continue

            # Make sure it is shaped as N x 2.
            region_pixels = region_pixels.reshape(-1, 2)

            xs = region_pixels[:, 0]
            ys = region_pixels[:, 1]

            x1 = int(xs.min())
            x2 = int(xs.max()) + 1
            y1 = int(ys.min())
            y2 = int(ys.max()) + 1

            covered[y1:y2, x1:x2] = True

            region_green = green_mask_2d[y1:y2, x1:x2]

            # print(
            #     "REGION",
            #     i,
            #     "bbox:",
            #     x1, y1, x2, y2,
            #     "size:",
            #     x2 - x1,
            #     "x",
            #     y2 - y1,
            #     "green pixels:",
            #     int(np.count_nonzero(region_green))
            # )

        uncovered_green = green_mask_2d & ~covered

        # print(
        #     "GREEN PIXELS OUTSIDE ALL REGION BBOXES:",
        #     int(np.count_nonzero(uncovered_green))
        # )

        # if np.any(uncovered_green):
        #     uy, ux = np.where(uncovered_green)

        #     print(
        #         "UNEXPECTED GREEN RANGE:",
        #         "x =", int(ux.min()), int(ux.max()),
        #         "y =", int(uy.min()), int(uy.max())
        #     )

        if len(regions) == 0 and not self.is_mosaic:
            print("No green regions detected!")
            return

        output_img_array = ori_array[0].copy()

        for region_counter, region in enumerate(regions, 1):
            # bounding_box = expand_bounding(ori, region)
            bounding_box = expand_bounding(
                ori,
                region,
                expand_factor=1.5
            )

            crop_img = ori.crop(bounding_box)
            # crop_img.show()
            #convert mask back to image
            mask_reshaped = mask[0,:,:,:] * 255.0
            mask_img = Image.fromarray(mask_reshaped.astype('uint8'))
            #resize the cropped images
            crop_img = crop_img.resize(
                (512, 512),
                resample=Image.NEAREST
            )

            crop_img_array = np.asarray(
                crop_img,
                dtype=np.float32
            ) / 255.0

            crop_img_array = np.expand_dims(
                crop_img_array,
                axis=0
            )

            #resize the mask images
            mask_img = mask_img.crop(bounding_box)
            mask_img = mask_img.resize(
                (512, 512),
                resample=Image.NEAREST
            )
            
            # Convert mask image to array
            mask_array = np.asarray(mask_img, dtype=np.float32) / 255.0

            # Add batch dimension
            mask_array = np.expand_dims(mask_array, axis=0)

            # Debug input
            # debug_input = np.squeeze(crop_img_array, axis=0) * 255.0

            # Image.fromarray(
            #     np.clip(debug_input, 0, 255).astype(np.uint8)
            # ).save("/tmp/debug_model_input.png")

            # Predict
            pred_img_array = self.model.predict(
                [crop_img_array, mask_array]
            )

            # print(
            #     "prediction:",
            #     pred_img_array.min(),
            #     pred_img_array.max(),
            #     pred_img_array.mean()
            # )

            # Since final layer is sigmoid
            pred_img_array = np.squeeze(
                pred_img_array,
                axis=0
            )

            # print(
            #     "PRED GREEN:",
            #     np.min(pred_img_array[:, :, 1]),
            #     np.max(pred_img_array[:, :, 1]),
            #     "PRED RED:",
            #     np.min(pred_img_array[:, :, 0]),
            #     np.max(pred_img_array[:, :, 0]),
            #     "PRED BLUE:",
            #     np.min(pred_img_array[:, :, 2]),
            #     np.max(pred_img_array[:, :, 2])
            # )

            pred_green_dominant = (
                (pred_img_array[:, :, 1] > pred_img_array[:, :, 0] + 0.05) &
                (pred_img_array[:, :, 1] > pred_img_array[:, :, 2] + 0.05)
            )

            # print(
            #     "PRED GREEN-DOMINANT PIXELS:",
            #     np.sum(pred_green_dominant),
            #     "of",
            #     pred_img_array.shape[0] * pred_img_array.shape[1]
            # )

            pred_img_array = (
                np.clip(pred_img_array, 0, 1) * 255.0
            ).astype(np.uint8)

            #scale prediction image back to original size
            bounding_width = bounding_box[2]-bounding_box[0]
            bounding_height = bounding_box[3]-bounding_box[1]
            #convert np array to image

            # print(bounding_width,bounding_height)
            # print(pred_img_array.shape)

            pred_img = Image.fromarray(pred_img_array.astype('uint8'))
            # pred_img.show()
            pred_img = pred_img.resize((bounding_width, bounding_height), resample=Image.BICUBIC)
            pred_img_array = np.asarray(pred_img)
            pred_img_array = pred_img_array / 255.0

            # print(pred_img_array.shape)
            pred_img_array = np.expand_dims(pred_img_array, axis = 0)

            for i in range(len(ori_array)):
                for col in range(bounding_width):
                    for row in range(bounding_height):
                        x = col + bounding_box[0]
                        y = row + bounding_box[1]

                        if mask[0, y, x, 0] == 0:
                            output_img_array[y, x] = pred_img_array[i, row, col]

            print("{region_counter} out of {region_count} regions decensored.".format(region_counter=region_counter, region_count=len(regions)))

        output_img_array = output_img_array * 255.0

        #restore the alpha channel
        if has_alpha:
            #print(output_img_array.shape)
            #print(alpha_channel.shape)
            output_img_array = np.concatenate((output_img_array, alpha_channel), axis = 2)

        output_img = Image.fromarray(output_img_array.astype('uint8'))

        final_array = np.asarray(output_img).copy()

        remaining_green = np.all(
            final_array[:, :, :3] == [0, 255, 0],
            axis=2
        )

        remaining_count = int(np.count_nonzero(remaining_green))

        # print("FINAL EXACT GREEN PIXELS:", remaining_count)

        # if remaining_count > 0:
        #     ry, rx = np.where(remaining_green)

        #     print(
        #         "REMAINING GREEN RANGE:",
        #         "x =", int(rx.min()), int(rx.max()),
        #         "y =", int(ry.min()), int(ry.max())
        #     )

        #save the decensored image
        #file_name, _ = os.path.splitext(file_name)
        save_path = os.path.join(self.args.decensor_output_path, file_name)
        output_img.save(save_path)

        print("Decensored image saved to {save_path}!".format(save_path=save_path))
        return

if __name__ == '__main__':
    decensor = Decensor()
    decensor.decensor_all_images_in_folder()