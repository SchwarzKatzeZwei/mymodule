import os
from datetime import datetime as dt

from tensorflow_core.python.keras.callbacks import (EarlyStopping,
                                                    LambdaCallback,
                                                    LearningRateScheduler,
                                                    ModelCheckpoint,
                                                    TensorBoard)


class KerasCallBack:
    """Kerasコールバックラッパークラス"""
    @staticmethod
    def EarlyStoppingCallback(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False):
        """EarlyStopping Callback
            監視対象の数量が改善しなくなったら、トレーニングを停止します。

        Args:
            monitor (str, optional): 監視する数量. Defaults to 'val_loss'.
            min_delta (int, optional): 改善と見なされる監視対象数量の最小変化. Defaults to 0.
            patience (int, optional): トレーニングが停止されるまでの改善のないエポックの数. Defaults to 0.
            verbose (int, optional): 冗長モード. Defaults to 0.
            mode (str, optional): {"auto", "min", "max"}. Defaults to 'auto'.
            baseline ([type], optional): モニターされた数量のベースライン値. Defaults to None.
            restore_best_weights (bool, optional): モニターされた数量の最良の値を使用して、エポックからモデルの重みを復元するかどうか. Defaults to False.
        """
        return EarlyStopping(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode, baseline=baseline,
                             restore_best_weights=restore_best_weights)

    @staticmethod
    def TensorBoardCallBack(log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch', profile_batch=2,
                            embeddings_freq=0, embeddings_metadata=None, **kwargs):
        """TensorBoard Callback
            TensorBoardの視覚化を有効にします

        Args:
            log_dir (str, optional): TensorBoardによって解析されるログファイルを保存するディレクトリのパス. Defaults to 'logs'.
            histogram_freq (int, optional): モデルのレイヤーのアクティブ化ヒストグラムと重みヒストグラムを計算する頻度（エポック単位）. Defaults to 0.
            write_graph (bool, optional): TensorBoardでグラフを視覚化するかどうか. Defaults to True.
            write_images (bool, optional): モデルの重みを記述してTensorBoardで画像として視覚化するかどうか. Defaults to False.
            update_freq (str, optional): 'batch'または'epoch'または整数. Defaults to 'epoch'.
            profile_batch (int, optional): バッチをプロファイリングして、計算特性をサンプリングします. Defaults to 2.
            embeddings_freq (int, optional): 埋め込みレイヤーが視覚化される頻度（エポック単位）. Defaults to 0.
            embeddings_metadata ([type], optional): レイヤー名をこの埋め込みレイヤーのメタデータが保存されているファイル名にマップする辞書. Defaults to None.
        """
        tdatetime = dt.now()
        tstr = tdatetime.strftime('%Y%m%d-%H%M%S')
        return TensorBoard(log_dir=log_dir + os.sep + tstr, histogram_freq=histogram_freq, write_graph=write_graph, write_images=write_images,
                           update_freq=update_freq, profile_batch=profile_batch, embeddings_freq=embeddings_freq,
                           embeddings_metadata=embeddings_metadata, **kwargs)

    @staticmethod
    def LambdaCallBack(on_epoch_begin=None, on_epoch_end=None, on_batch_begin=None, on_batch_end=None, on_train_begin=None, on_train_end=None, **kwargs):
        """Lambda Callback
            オンザフライでシンプルなカスタムコールバックを作成するためのコールバック
        Args:
            on_epoch_begin: すべてのエポックの開始時に呼ばれます．lambda epoch, logs: print(epoch, logs)
            on_epoch_end: すべてのエポックの終了時に呼ばれます．  lambda epoch, logs: print(epoch, logs)
            on_batch_begin: すべてのバッチの開始時に呼ばれます．  lambda batch, logs: print(batch, logs)
            on_batch_end: すべてのバッチの終了時に呼ばれます．    lambda batch, logs: print(batch, logs)
            on_train_begin: 訓練の開始時に呼ばれます．            lambda logs: print(logs)
            on_train_end: 訓練の終了時に呼ばれます．              lambda logs: print(logs)
        """
        return LambdaCallback(on_epoch_begin=on_epoch_begin, on_epoch_end=on_epoch_end, on_batch_begin=on_batch_begin, on_batch_end=on_batch_end,
                              on_train_begin=on_train_begin, on_train_end=on_train_end)

    @staticmethod
    def ModelCheckpointCallBack(filepath, monitor='val_loss', verbose=0, save_best_only=False, save_weights_only=False, mode='auto', save_freq='epoch',
                                **kwargs):
        """ModelCheckpoint Callback
            エポックごとにモデルを保存します

        Args:
            filepath (str): モデルファイルを保存するパス
            monitor (str, optional): 監視する数量. Defaults to 'val_loss'.
            verbose (int, optional): 詳細モード、0または1. Defaults to 0.
            save_best_only (bool, optional): Trueの場合監視数量に応じた最新の最良モデルは上書きされません. Defaults to False.
            save_weights_only (bool, optional): Trueの場合、モデルのウェイトのみmodel.save_weights(filepath)が保存されます（model.save(filepath)）. Defaults to False.
            mode (str, optional): {auto、min、max}. Defaults to 'auto'.
            save_freq (str, optional): 'epoch'または整数. Defaults to 'epoch'.
        """
        return ModelCheckpoint(filepath=filepath, monitor=monitor, verbose=verbose, save_best_only=save_best_only, save_weights_only=save_weights_only,
                               mode=mode, save_freq=save_freq, **kwargs)

    def LearningRateSchedulerCallBack(self, schedule, verbose=0):
        """LearningRateScheduler CallBack
            学習率スケジューラ
        Args:
            schedule ([type]): エポックインデックスを入力として受け取り（整数、0からインデックス付け）、新しい学習率を出力（float）として返す関数
            verbose (int, optional): 0：何も表示しない、1：メッセージを更新します. Defaults to 0.
        """
        return LearningRateScheduler(schedule=schedule, verbose=verbose)
