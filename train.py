import argparse
from pathlib import Path
import tensorflow as tf

import config
import dataset
import model as unet_module
import utils


def train(
    use_nir: bool = True,
    samples_per_class: int = 512,
    epochs: int = config.EPOCHS,
    batch_size: int = config.BATCH_SIZE,
    learning_rate: float = config.LEARNING_RATE,
    resume: bool = True
):
    print("=" * 60)
    print(f"Starting Training - Mode: {'RGB + NIR (4 channels)' if use_nir else 'RGB (3 channels)'}")
    print("=" * 60)

    # 1. Dataset verification
    if not utils.verify_dataset():
        print("Cannot start training without dataset.")
        return

    # 2. File preparation
    print("Scanning dataset files...")
    train_files = dataset.get_image_files(config.TRAIN_RGB_DIR)
    val_files = dataset.get_image_files(config.VAL_RGB_DIR)

    print(f"Found {len(train_files)} train files and {len(val_files)} val files.")

    print("Balancing training dataset...")
    pos_files, neg_files = dataset.find_positive_negative_files(
        train_files, config.TRAIN_LABEL_DIR
    )
    print(f"Positive samples: {len(pos_files)}, Negative samples: {len(neg_files)}")

    balanced_train_files = dataset.create_balanced_filelist(
        pos_files, neg_files, samples_per_class=samples_per_class
    )
    print(f"Selected balanced training set: {len(balanced_train_files)} files.")

    # 3. Create tf.data pipelines
    nir_dir = config.TRAIN_NIR_DIR if use_nir else None
    val_nir_dir = config.VAL_NIR_DIR if use_nir else None

    train_ds = dataset.make_dataset(
        balanced_train_files,
        rgb_dir=config.TRAIN_RGB_DIR,
        label_dir=config.TRAIN_LABEL_DIR,
        nir_dir=nir_dir,
        batch_size=batch_size,
        training=True
    )

    val_ds = dataset.make_dataset(
        val_files,
        rgb_dir=config.VAL_RGB_DIR,
        label_dir=config.VAL_LABEL_DIR,
        nir_dir=val_nir_dir,
        batch_size=batch_size,
        training=False
    )

    # 4. Model Setup & Resume capability
    in_channels = 4 if use_nir else 3
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, in_channels)
    model_save_path = config.RGB_NIR_MODEL_PATH if use_nir else config.RGB_MODEL_PATH
    config.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    custom_objects = {
        "combined_loss": unet_module.combined_loss,
        "dice_loss": unet_module.dice_loss
    }

    if resume and model_save_path.exists():
        print(f"Found existing checkpoint at {model_save_path}. Resuming training...")
        try:
            model = tf.keras.models.load_model(
                model_save_path,
                custom_objects=custom_objects
            )
        except Exception as e:
            print(f"Could not load checkpoint ({e}). Building fresh model...")
            model = unet_module.build_unet(input_shape=input_shape)
            model = unet_module.compile_unet(model, learning_rate=learning_rate)
    else:
        print("Building fresh model...")
        model = unet_module.build_unet(input_shape=input_shape)
        model = unet_module.compile_unet(model, learning_rate=learning_rate)

    model.summary()

    # 5. Checkpoints and Callbacks
    best_checkpoint_path = config.CHECKPOINT_DIR / f"{'rgb_nir' if use_nir else 'rgb'}_best.keras"
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(best_checkpoint_path),
            monitor="val_loss",
            save_best_only=True,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(model_save_path),
            save_best_only=False,
            verbose=0
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=6,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=3,
            verbose=1
        )
    ]

    print(f"\nStarting model fit for {epochs} epochs...")
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=callbacks,
        verbose=1
    )

    print(f"Training completed! Model saved to {model_save_path} and best weights to {best_checkpoint_path}")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for Crop Stress (SIH25099)")
    parser.add_argument("--nir", action="store_true", default=True, help="Use 4-channel RGB+NIR")
    parser.add_argument("--rgb-only", action="store_true", help="Train on 3-channel RGB only")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--samples", type=int, default=512, help="Samples per class for balancing")
    parser.add_argument("--fresh", action="store_true", help="Start training fresh without loading checkpoint")

    args = parser.parse_args()
    use_nir = False if args.rgb_only else args.nir
    train(
        use_nir=use_nir,
        samples_per_class=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=(not args.fresh)
    )

