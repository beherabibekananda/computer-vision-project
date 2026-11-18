from typing import Tuple
import tensorflow as tf
from tensorflow.keras import layers, models


def build_unet(input_shape: Tuple[int, int, int] = (256, 256, 3)) -> models.Model:
    """Builds a U-Net architecture adaptable to 3-channel (RGB) or 4-channel (RGB+NIR)."""
    inputs = layers.Input(shape=input_shape, name="input_image")

    # ENCODER - Block 1
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same", name="enc_c1_1")(inputs)
    c1 = layers.Conv2D(32, 3, activation="relu", padding="same", name="enc_c1_2")(c1)
    p1 = layers.MaxPooling2D((2, 2), name="enc_pool1")(c1)

    # ENCODER - Block 2
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_c2_1")(p1)
    c2 = layers.Conv2D(64, 3, activation="relu", padding="same", name="enc_c2_2")(c2)
    p2 = layers.MaxPooling2D((2, 2), name="enc_pool2")(c2)

    # ENCODER - Block 3
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same", name="enc_c3_1")(p2)
    c3 = layers.Conv2D(128, 3, activation="relu", padding="same", name="enc_c3_2")(c3)
    p3 = layers.MaxPooling2D((2, 2), name="enc_pool3")(c3)

    # BOTTLENECK
    b = layers.Conv2D(256, 3, activation="relu", padding="same", name="bottleneck_1")(p3)
    b = layers.Conv2D(256, 3, activation="relu", padding="same", name="bottleneck_2")(b)

    # DECODER - Block 3
    u3 = layers.UpSampling2D((2, 2), name="dec_up3")(b)
    u3 = layers.Concatenate(name="dec_concat3")([u3, c3])
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same", name="dec_c4_1")(u3)
    c4 = layers.Conv2D(128, 3, activation="relu", padding="same", name="dec_c4_2")(c4)

    # DECODER - Block 2
    u2 = layers.UpSampling2D((2, 2), name="dec_up2")(c4)
    u2 = layers.Concatenate(name="dec_concat2")([u2, c2])
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same", name="dec_c5_1")(u2)
    c5 = layers.Conv2D(64, 3, activation="relu", padding="same", name="dec_c5_2")(c5)

    # DECODER - Block 1
    u1 = layers.UpSampling2D((2, 2), name="dec_up1")(c5)
    u1 = layers.Concatenate(name="dec_concat1")([u1, c1])
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same", name="dec_c6_1")(u1)
    c6 = layers.Conv2D(32, 3, activation="relu", padding="same", name="dec_c6_2")(c6)

    # OUTPUT: 1 channel binary probability map
    outputs = layers.Conv2D(1, 1, activation="sigmoid", name="output_mask")(c6)

    model = models.Model(inputs=inputs, outputs=outputs, name=f"unet_{input_shape[-1]}ch")
    return model


def dice_loss(y_true, y_pred, smooth: float = 1e-6):
    """Sørensen–Dice loss for binary segmentation."""
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    axes = [1, 2, 3]
    intersection = tf.reduce_sum(y_true * y_pred, axis=axes)
    denominator = tf.reduce_sum(y_true, axis=axes) + tf.reduce_sum(y_pred, axis=axes)

    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - tf.reduce_mean(dice)


def combined_loss(y_true, y_pred):
    """Combined Binary Cross-Entropy and Dice Loss."""
    bce = tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))
    d_loss = dice_loss(y_true, y_pred)
    return bce + d_loss


def compile_unet(model: models.Model, learning_rate: float = 1e-4) -> models.Model:
    """Compiles the model with Adam, combined loss, and binary accuracy."""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=combined_loss,
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="binary_accuracy")
        ]
    )
    return model

