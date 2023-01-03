from keras.callbacks import TensorBoard
import tensorflow as tf2
import os


class ModifiedTensorBoard(TensorBoard):
    def __init__(self, name, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf2.summary.create_file_writer(os.path.join(self.log_dir, name))
        self._log_write_dir = os.path.join(self.log_dir, name)

    def set_model(self, model):
        self.model = model

        self._train_dir = os.path.join(self._log_write_dir, 'train')
        self._train_step = self.model._train_counter

        self._val_dir = os.path.join(self._log_write_dir, 'validation')
        self._val_step = self.model._test_counter

        self._should_write_train_graph = False

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf2.summary.scalar(key, value, step = self.step)
                self.writer.flush()