
import logging
import tensorflow as tf
import os

# Configure your custom logger
logger = logging.getLogger(__name__)

# Redirect TensorFlow logs to your logger
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR

class TFLoggerAdapter(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logger.info(f"Epoch {epoch+1} completed - loss: {logs.get('loss'):.4f}, "
                   f"accuracy: {logs.get('accuracy'):.4f}, "
                   f"val_loss: {logs.get('val_loss'):.4f}, "
                   f"val_accuracy: {logs.get('val_accuracy'):.4f}")

    def on_train_begin(self, logs=None):
        logger.info("Training started")

    def on_train_end(self, logs=None):
        logger.info("Training completed")

# Use in model training
# model.fit(..., callbacks=[TFLoggerAdapter()])

