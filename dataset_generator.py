# -*- coding: utf-8 -*-
import os
import glob
import argparse
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
from lib.image_data_generator_custom import ImageDataGeneratorCustom


def getParse():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        help="source directory",
                        required=True,
                        type=str)
    parser.add_argument("-d",
                        help="destination directory",
                        required=True,
                        type=str)
    parser.add_argument("-m",
                        help="生成掛け数",
                        required=False,
                        default=1,
                        type=int)
    args = parser.parse_args()
    return args


def main(args):
    os.makedirs(args.d, exist_ok=True)

    # 拡張する際の設定
    # https://keras.io/ja/preprocessing/image/#imagedatagenerator_1
    generator = ImageDataGeneratorCustom(
        # random_crop=[900, 900],
        mix_up_alpha=2,
        # cutout_mask_size=100
    )

    for i, img_path in enumerate(glob.glob(args.s+"/*")):
        img = load_img(img_path)
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)

        for idx, d in enumerate(generator.flow(x, [i], batch_size=1)):
            dg_img = array_to_img(d[0][0], scale=True)
            filename = args.d + "/" + os.path.splitext(os.path.basename(img_path))[0] \
                + '_aug-' + str(idx).zfill(3) + ".jpg"
            dg_img.save(filename)
            if args.m < idx + 1:
                break

    # Mix-up使う場合（確認用）
    imgs = []
    for i, img_path in enumerate(glob.glob(args.s+"/*")):
        img = load_img(img_path)
        img = img.resize((900, 900))
        x = img_to_array(img)
        imgs.append(np.asarray(x))
    imgs = np.array(imgs)
    for idx, d in enumerate(generator.flow(imgs, [0, 1], batch_size=32)):
        dg_img = array_to_img(d[0][0], scale=True)
        filename = args.d + "/" + os.path.splitext(os.path.basename(img_path))[0] \
            + '_aug-' + str(idx).zfill(3) + ".jpg"
        dg_img.save(filename)
        if args.m < idx + 1:
            break


if __name__ == "__main__":
    args = getParse()
    main(args)
