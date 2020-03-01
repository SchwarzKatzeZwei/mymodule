import cv2
import numpy as np


class OpenCVOrgUtil:
    """OpenCVユーティリティ"""

    @staticmethod
    def get_height(img: np.ndarray) -> int:
        """画像の縦幅を返却する

        Args:
            img (numpy.ndarray): 画像データ

        Returns:
            int: 画像縦幅
        """
        return img.shape[0]

    @staticmethod
    def get_width(img: np.ndarray) -> int:
        """画像横幅を返却する

        Args:
            img (numpy.ndarray): 画像データ

        Returns:
            int: 画像横幅
        """
        return img.shape[1]

    @staticmethod
    def get_channel(img: np.ndarray) -> int:
        """画像チャンネル数を返却する

        Args:
            img (numpy.ndarray): 画像データ

        Returns:
            int: 画像チャンネル数
        """
        return img.shape[2]

    @staticmethod
    def erosion(img: np.ndarray, kernel: list, iterations=1) -> np.ndarray:
        """収縮処理

        Args:
            img (numpy.ndarray): 画像データ
            kernel (list[int, int]): カーネルサイズ
            iterations (int, optional): 試行回数 Defaults to 1

        Returns:
            numpy.ndarray: 処理画像
        """
        return cv2.erode(img, kernel, iterations=iterations)

    @staticmethod
    def dilation(img: np.ndarray, kernel: list, iterations=1) -> np.ndarray:
        """膨張処理

        Args:
            img (numpy.ndarray): 画像データ
            kernel (list(int, int)): カーネルサイズ
            iterations (int, optional): 試行回数 Defaults to 1

        Returns:
            numpy.ndarray: 処理画像
        """
        return cv2.dilate(img, kernel, iterations=iterations)

    @staticmethod
    def opening(img: np.ndarray, kernel: list, iterations=1) -> np.ndarray:
        """オープニング処理（収縮->膨張処理）

        Args:
            img (numpt.ndarray): 画像データ
            kernel (list(int, int)): カーネルサイズ
            iterations (int, optional): 試行回数 Defaults to 1

        Returns:
            numpy.ndarray: 処理画像
        """
        return cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)

    @staticmethod
    def closing(img: np.ndarray, kernel: list, iterations=1) -> np.ndarray:
        """クロージング処理（膨張->収縮処理）

        Args:
            img (numpy.ndarray): 画像データ
            kernel (list(int, int)): カーネルサイズ
            iterations (int, optional): 試行回数 Defaults to 1

        Returns:
            numpy.ndarray: 処理画像
        """
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)

    @staticmethod
    def gaussian_blur(img: np.ndarray, kernel: list, sigmaX=0) -> np.ndarray:
        """ガウシアンフィルタ（画像平滑化）

        Args:
            img (numpy.ndarray): 画像データ
            kernel (list(int, int)): カーネルサイズ
            sigma (int, optional): ガウシアンの標準偏差値横方向 Defaults to 0

        Returns:
            numpy.ndarray: 処理画像
        """
        return cv2.GaussianBlur(img, kernel, sigmaX)

    @staticmethod
    def contrast_correction(img: np.ndarray, contrast: float) -> np.ndarray:
        """コントラスト補正

        Args:
            img (numpy.ndarray): 画像データ(gray)
            contrast (float)): ガンマ値

        Returns:
            numpy.ndarray: 処理画像
        """
        imax = img.max()
        img = imax * (img / imax) ** (1 / contrast)
        return img.astype('uint8') * 255

    @staticmethod
    def gamma_correction(img: np.ndarray, gamma: float) -> np.ndarray:
        """ガンマ補正

        Args:
            img (numpy.ndarray): 画像データ
            gamma (float): ガンマ値

        Returns:
            numpy.ndarray: 処理画像
        """
        table = (np.arange(256) / 255) ** gamma * 255
        table = np.clip(table, 0, 255).astype(np.uint8)
        return cv2.LUT(img, table)

    @staticmethod
    def split_VH(img: np.ndarray, ver_num: int, hor_num: int) -> list:
        """画像を縦横指定分割する

        Args:
            img (numpy.ndarray): 画像データ
            ver_num (int): 縦分割数
            hor_num (int): 横分割数

        Returns:
            list[numpy.ndarray]: 分割画像リスト
        """
        results = []
        height, width = img.shape[0:2]
        vertical_position = 0
        for vertical in range(ver_num):
            horizontal_position = 0
            for horizontal in range(hor_num):
                x = (width // hor_num) * (horizontal + 1)
                y = (height // ver_num) * (vertical + 1)
                if horizontal + 1 == hor_num:
                    x += width % hor_num
                if vertical + 1 == ver_num:
                    y += height % ver_num

                results.append(img[vertical_position: y, horizontal_position: x])

                horizontal_position = (width // hor_num) * (horizontal + 1)
            vertical_position = (height // ver_num) * (vertical + 1)

        return results

    @staticmethod
    def concat_VH(imgs: list, ver_num: int, hor_num: int) -> np.ndarray:
        """画像を縦横指定結合する

        Args:
            imgs (list[numpy.ndarray]): 分割画像リスト
            ver_num (int): 縦分割数
            hor_num (int): 横分割数

        Returns:
            numpy.ndarray: 結合画像
        """
        assert ver_num >= 1
        assert hor_num >= 1
        if hor_num == 1 and ver_num == 1:
            return imgs[0]

        rows = []
        for idx in range(ver_num):
            rows.append(cv2.hconcat(imgs[hor_num * idx: hor_num * idx + hor_num]))
        concat_img = cv2.vconcat(rows)

        return concat_img

    @staticmethod
    def key_stop() -> None:
        """キー入力待ちを発生させる"""

        while True:
            key = cv2.waitKey(0) & 0xFF
            if key == ord("w"):
                cv2.destroyAllWindows()
                return "w"
            elif key == ord("n"):
                return "n"
            elif key == ord("q"):
                exit()

    @staticmethod
    def crop(img: np.ndarray, xmin: int, ymin: int, xmax: int, ymax: int) -> np.ndarray:
        """画像をクロップする

        Args:
            img (numpy.ndarray): 画像データ
            xmin (int): 最左座標
            ymin (int): 最上座標
            xmax (int): 最右座標
            ymax (int): 最下座標

        Returns:
            numpy.ndarray: クロップ画像
        """
        return img[ymin:ymax, xmin:xmax]

    @staticmethod
    def affin(image: np.ndarray, angle: float) -> np.ndarray:
        """画像を回転（アフィン変換）する

        Args:
            image (np.ndarray): 画像データ
            angle (float): 角度(時計回り)

        Returns:
            numpy.ndarray: 回転画像
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, -angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    @staticmethod
    def paste(bg_img: np.ndarray, fg_img: np.ndarray, x=0, y=0, alpha_bg=0.5, alpha_fg=0.5) -> np.ndarray:
        """画像に別の画像画像を貼り付ける。

        Args:
            bg_img (numpy.ndarray): 背景となる画像データ
            fg_img (numpy.ndarray): 前傾となる画像データ
            x (int, optional): 貼付位置X座標 Defaults to 0
            y (int, optional): 貼付位置Y座標 Defaults to 0
            alpha_bg (float, optional): 背景画像アルファ値 Defaults to 0.5
            alpha_fg (float, optional): 前景画像アルファ値 Defaults to 0.5

        Returns:
            numpy.ndarray: 合成画像
        """
        (top_h, top_w) = fg_img.shape[:2]
        bg_img[y:y+top_h, x:x+top_w] = cv2.addWeighted(bg_img[y:y+top_h, x:x+top_w], alpha_bg, fg_img, alpha_fg, 0)
        return bg_img

    @staticmethod
    def scale_to_width(img: np.ndarray, width: int) -> np.ndarray:
        """width指定リサイズ

        Args:
            img (numpy.ndarray): 画像データ
            width (int): 指定横幅

        Returns:
            numpy.ndarray: 処理画像
        """
        scale = width / img.shape[1]
        return cv2.resize(img, dsize=None, fx=scale, fy=scale)

    @staticmethod
    def scale_to_height(img: np.ndarray, height: int) -> np.ndarray:
        """height指定リサイズ

        Args:
            img (numpy.ndarray): 画像データ
            height (int): 指定縦幅

        Returns:
            numpy.ndarray: 処理画像
        """
        scale = height / img.shape[0]
        return cv2.resize(img, dsize=None, fx=scale, fy=scale)

    @staticmethod
    def template_match(temple_img: np.ndarray, target_img: np.ndarray) -> float:
        """テンプレートマッチングを実施する

        Args:
            temple_img (numpy.ndarray): テンプレート画像
            target_img (numpy.ndarray): 比較対象画像

        Returns:
            float: 類似度
        """
        src_h, src_w = temple_img.shape[:2]
        dst_h, dst_w = target_img.shape[:2]
        if (src_h - dst_h) * (src_w - dst_w) < 0:
            return 0
        else:
            if (src_h > dst_h) or (src_w > dst_w):
                src = temple_img
                dst = target_img
            else:
                src = target_img
                dst = temple_img
            res = cv2.matchTemplate(dst, src, cv2.TM_CCOEFF_NORMED)
            return round(np.max(res), 2)

    @staticmethod
    def hists_match(src_img: np.ndarray, dst_img: np.ndarray) -> float:
        """ヒストグラムマッチングを実施する

        http://pynote.hatenablog.com/entry/opencv-comparehist

        Args:
            src_img (numpy.ndarray): テンプレート画像
            dst_img (numpy.ndarray): 比較対象画像

        Returns:
            float: 類似度
        """
        ch_names = {0: "Hue", 1: "Saturation", 2: "Brightness"}

        # HSV に変換する。
        hsv1 = cv2.cvtColor(src_img, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(dst_img, cv2.COLOR_BGR2HSV)

        # 各チャンネルごとにヒストグラムの類似度を算出する。
        scores = []
        for ch in ch_names:
            h1 = cv2.calcHist([hsv1], [ch], None, histSize=[256], ranges=[0, 256])
            h2 = cv2.calcHist([hsv2], [ch], None, histSize=[256], ranges=[0, 256])
            score = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)
            scores.append(score)
        mean = np.mean(scores)
        return round(mean, 2)

    @staticmethod
    def akaze_match(src_img: np.ndarray, dst_img: np.ndarray) -> float:
        """AKAZE特徴量マッチングを実施する
        https://qiita.com/hitomatagi/items/caac014b7ab246faf6b1

        Args:
            src_img (numpy.ndarray): テンプレート画像
            dst_img (numpy.ndarray): 比較対象画像

        Returns:
            float: 特徴量の距離
        """
        # A-KAZE検出器の生成
        akaze = cv2.AKAZE_create()

        # 特徴量の検出と特徴量ベクトルの計算
        kp1, des1 = akaze.detectAndCompute(src_img, None)
        kp2, des2 = akaze.detectAndCompute(dst_img, None)

        # Brute-Force Matcher生成
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)

        try:
            # 特徴量ベクトル同士をBrute-Force＆KNNでマッチング
            matches = bf.knnMatch(des1, des2, k=2)
            distance = [m.distance for m, n in matches]
            return round(sum(distance) / len(distance), 2)
        except Exception as e:
            print(e)
            return 99999
