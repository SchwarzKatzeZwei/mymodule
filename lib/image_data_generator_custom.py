import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator


class ImageDataGeneratorCustom(ImageDataGenerator):
    def __init__(self, featurewise_center=False, samplewise_center=False,
                 featurewise_std_normalization=False, samplewise_std_normalization=False,
                 zca_whitening=False, zca_epsilon=1e-06, rotation_range=0.0, width_shift_range=0.0,
                 height_shift_range=0.0, brightness_range=None, shear_range=0.0, zoom_range=0.0,
                 channel_shift_range=0.0, fill_mode='nearest', cval=0.0, horizontal_flip=False,
                 vertical_flip=False, rescale=None, preprocessing_function=None, data_format=None, validation_split=0.0,
                 random_crop=None, mix_up_alpha=0.0, cutout_mask_size=0):
        # 親クラスのコンストラクタ
        super().__init__(featurewise_center, samplewise_center, featurewise_std_normalization,
                         samplewise_std_normalization, zca_whitening, zca_epsilon, rotation_range, width_shift_range,
                         height_shift_range, brightness_range, shear_range, zoom_range, channel_shift_range, fill_mode,
                         cval, horizontal_flip, vertical_flip, rescale, preprocessing_function, data_format,
                         validation_split)
        # 拡張処理のパラメーター
        # Mix-up
        assert mix_up_alpha >= 0.0
        self.mix_up_alpha = mix_up_alpha
        # Random Crop
        assert random_crop is None or len(random_crop) == 2
        self.random_crop_size = random_crop
        # CutOut
        assert cutout_mask_size >= 0
        self.cutout_mask_size = cutout_mask_size

    # ランダムクロップ
    # https://jkjung-avt.github.io/keras-image-cropping/
    def random_crop(self, original_img):
        # Note: image_data_format is 'channel_last'
        assert original_img.shape[2] == 3
        if original_img.shape[0] < self.random_crop_size[0] or original_img.shape[1] < self.random_crop_size[1]:
            raise ValueError(
                f"Invalid random_crop_size : original = {original_img.shape}, crop_size = {self.random_crop_size}")

        height, width = original_img.shape[0], original_img.shape[1]
        dy, dx = self.random_crop_size
        x = np.random.randint(0, width - dx + 1)
        y = np.random.randint(0, height - dy + 1)
        return original_img[y:(y+dy), x:(x+dx), :]

    # Mix-up
    # 参考 https://qiita.com/yu4u/items/70aa007346ec73b7ff05
    def mix_up(self, X1, y1, X2, y2):
        assert X1.shape[0] == y1.shape[0] == X2.shape[0] == y2.shape[0]
        batch_size = X1.shape[0]
        lb = np.random.beta(self.mix_up_alpha, self.mix_up_alpha, batch_size)
        X_l = lb.reshape(batch_size, 1, 1, 1)
        y_l = lb.reshape(batch_size, 1)
        X = X1 * X_l + X2 * (1-X_l)
        y = y1 * y_l + y2 * (1 - y_l)
        return X, y

    def cutout(self, x, y):
        return np.array(list(map(self._cutout, x))), y

    def _cutout(self, image_origin):
        # 最後に使うfill()は元の画像を書き換えるので、コピーしておく
        image = np.copy(image_origin)
        mask_value = image.mean()

        h, w, _ = image.shape
        # マスクをかける場所のtop, leftをランダムに決める
        # はみ出すことを許すので、0以上ではなく負の値もとる(最大mask_size // 2はみ出す)
        top = np.random.randint(0 - self.cutout_mask_size // 2, h - self.cutout_mask_size)
        left = np.random.randint(0 - self.cutout_mask_size // 2, w - self.cutout_mask_size)
        bottom = top + self.cutout_mask_size
        right = left + self.cutout_mask_size

        # はみ出した場合の処理
        if top < 0:
            top = 0
        if left < 0:
            left = 0

        # マスク部分の画素値を平均値で埋める
        image[top:bottom, left:right, :].fill(mask_value)
        return image

    def flow_from_directory(self, directory, target_size=(256, 256), color_mode='rgb',
                            classes=None, class_mode='categorical', batch_size=32, shuffle=True,
                            seed=None, save_to_dir=None, save_prefix='', save_format='png',
                            follow_links=False, subset=None, interpolation='nearest'):
        # 親クラスのflow_from_directory
        batches = super().flow_from_directory(directory, target_size, color_mode, classes, class_mode, batch_size,
                                              shuffle, seed, save_to_dir, save_prefix, save_format, follow_links,
                                              subset, interpolation)
        # 拡張処理
        while True:
            batch_x, batch_y = next(batches)
            if self.mix_up_alpha > 0:
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                batch_x, batch_y = self.mix_up(
                    batch_x, batch_y, batch_x_2, batch_y_2)

            # Random crop
            if self.random_crop_size is not None:
                x = np.zeros(
                    (batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 3))
                for i in range(batch_x.shape[0]):
                    x[i] = self.random_crop(batch_x[i])
                batch_x = x

            # CutOut
            if self.cutout_mask_size > 0:
                result = self.cutout(batch_x, batch_y)
                batch_x, batch_y = result

            # 返り値
            yield (batch_x, batch_y)

    def flow(self, *args, **kwargs):
        batches = super().flow(*args, **kwargs)
        # 拡張処理
        while True:
            batch_x, batch_y = next(batches)

            # Mix-up
            if self.mix_up_alpha > 0:
                while True:
                    batch_x_2, batch_y_2 = next(batches)
                    m1, m2 = batch_x.shape[0], batch_x_2.shape[0]
                    if m1 < m2:
                        batch_x_2 = batch_x_2[:m1]
                        batch_y_2 = batch_y_2[:m1]
                        break
                    elif m1 == m2:
                        break
                batch_x, batch_y = self.mix_up(
                    batch_x, batch_y, batch_x_2, batch_y_2)

            # Random crop
            if self.random_crop_size is not None:
                x = np.zeros(
                    (batch_x.shape[0], self.random_crop_size[0], self.random_crop_size[1], 3))
                for i in range(batch_x.shape[0]):
                    x[i] = self.random_crop(batch_x[i])
                batch_x = x

            # CutOut
            if self.cutout_mask_size > 0:
                result = self.cutout(batch_x, batch_y)
                batch_x, batch_y = result

            # 返り値
            yield (batch_x, batch_y)
