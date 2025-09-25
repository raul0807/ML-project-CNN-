import os, sys, time, random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

from sklearn.model_selection import StratifiedKFold, train_test_split, ParameterGrid
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# ---------------- CONFIG ----------------
# define main paths and training/testing setting
DEFAULT_PARENT = "/Users/raulspano/Desktop/archive"
TRAIN_SUBFOLDER = "rps-cv-images"
TEST_SUBFOLDER = "rps-test-set"    # optional, if available
IMG_SIZE = (128, 128)              # image resize shape
SEED = 42                          # random seed for reproducibility
np.random.seed(SEED); random.seed(SEED); tf.random.set_seed(SEED)

# Grid search and Cross-Validations settings
K_FOLDS = 4             # number of folds for cross-validation (5 ideal, 4 faster)
EPOCHS_CV = 6           # epochs during cross-validation (low for speed)
EPOCHS_FINAL = 12       # epochs for final retraining
PATIENCE = 3            # patience for early stopping

# hyperparameter grid for tuning
PARAM_GRID = {
    'lr': [1e-3, 5e-4],
    'batch_size': [32],   # we can extend with[32,64] but increasws runtime
    'dropout': [0.25, 0.5]
}

# ---------------- utilities ----------------
def find_dirs(parent=DEFAULT_PARENT):
    # locate training and test directories inside the dateset parent folder
    train_dir = os.path.join(parent, TRAIN_SUBFOLDER)
    test_dir = os.path.join(parent, TEST_SUBFOLDER)
    test_dir = test_dir if os.path.isdir(test_dir) else None
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"Train dir non trovato: {train_dir}")
    return train_dir, test_dir

def load_images_from_folder(folder, img_size=IMG_SIZE):
    # load all images from a given folder and its subfolders, each subfolder represents a class label
    # returns arrays X (images), y (labels), class names, and filenames
    X, y, filenames = [], [], []
    classes = sorted([d for d in os.listdir(folder) if os.path.isdir(os.path.join(folder,d))])
    cls2idx = {c:i for i,c in enumerate(classes)}
    for c in classes:
        cdir = os.path.join(folder, c)
        for fname in sorted(os.listdir(cdir)):
            if fname.lower().endswith(('.png','.jpg','.jpeg','.bmp')):
                path = os.path.join(cdir, fname)
                try:
                    img = Image.open(path).convert('RGB').resize(img_size)
                    arr = np.asarray(img, dtype=np.float32) / 255.0
                    X.append(arr)
                    y.append(cls2idx[c])
                    filenames.append(path)
                except Exception as e:
                    print("Impossibile aprire:", path, e)
    X = np.stack(X, axis=0)
    y = np.array(y, dtype=np.int32)
    return X, y, classes, filenames

# ---------------- models ----------------
def build_model_A(input_shape=IMG_SIZE+(3,), num_classes=3, dropout=0.25):
    # Model A simple CNN with two convolutional layers, followed by dense layers and dropout
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16,3,activation='relu',padding='same')(inputs)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(64,activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs,out)

def build_model_B(input_shape=IMG_SIZE+(3,), num_classes=3, dropout=0.3):
    # Model B deeper CNN with more filters, dropout layers and global average pooling before dense layers
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128,activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    return keras.Model(inputs,out)

def build_model_C(input_shape=IMG_SIZE+(3,), num_classes=3, dropout=0.5):
    # Model C more complex CNN with batch normalization, multiple convolutional layers and higher capacity dense layer
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(inputs)
    x = layers.Conv2D(32,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.Conv2D(64,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(128,3,activation='relu',padding='same')(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    out = layers.Dense(3, activation='softmax')(x)
    return keras.Model(inputs,out)

# ---------------- grid-search + kfold ----------------
def run_grid_cv(X_cv, y_cv, build_fn, param_grid, model_label):
    # perform grid search with cross validation
    results = []
    param_list = list(ParameterGrid(param_grid)) # iterates over parameter grid
    skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=SEED) # perform stratified K-folds
    # trains model for each fold, collects accuracy scores and returns sorted results by mean accuracy
    for p_i, params in enumerate(param_list):
        print(f"\n--- {model_label} Grid {p_i+1}/{len(param_list)}: {params} ---")
        fold_scores = []
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv), start=1):
            print(f" Fold {fold}/{K_FOLDS} ...", end=' ')
            X_train, X_val = X_cv[train_idx], X_cv[val_idx]
            y_train_int, y_val_int = y_cv[train_idx], y_cv[val_idx]
            y_train = to_categorical(y_train_int, num_classes=np.max(y_cv)+1)
            y_val = to_categorical(y_val_int, num_classes=np.max(y_cv)+1)

            # data augmentation for training set
            train_datagen = ImageDataGenerator(
                rotation_range=15, width_shift_range=0.08, height_shift_range=0.08,
                shear_range=0.05, zoom_range=0.08, horizontal_flip=True, fill_mode='nearest'
            )
            val_datagen = ImageDataGenerator()

            train_gen = train_datagen.flow(X_train, y_train, batch_size=params['batch_size'], shuffle=True, seed=SEED)
            val_gen = val_datagen.flow(X_val, y_val, batch_size=params['batch_size'], shuffle=False)

            # build model with current hyperparametres
            model = build_fn(dropout=params['dropout'])
            model.compile(optimizer=keras.optimizers.Adam(learning_rate=params['lr']),
                          loss='categorical_crossentropy', metrics=['accuracy'])
            # callbacks: early stopping to prevent overfitting
            cb = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)]
            history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_CV, callbacks=cb, verbose=0)
            # evaluation on validation set
            _, acc = model.evaluate(val_gen, verbose=0)
            print(f"val_acc={acc:.4f}")
            fold_scores.append(acc)
            # free GPU memory 
            keras.backend.clear_session()

        mean_acc = np.mean(fold_scores)
        std_acc = np.std(fold_scores)
        print(f" -> params {params} mean_acc={mean_acc:.4f} std={std_acc:.4f}")
        results.append({'params': params, 'mean_acc': mean_acc, 'std_acc': std_acc, 'folds': fold_scores})
    # sort results by mean accuracy 
    results_sorted = sorted(results, key=lambda r: r['mean_acc'], reverse=True)
    return results_sorted

# ---------------- retrain final on X_cv and evaluate on X_test ----------------
def retrain_and_evaluate(X_cv, y_cv, X_test, y_test, best_params, build_fn, model_label):
   # retrain the model on the full CV dataset with the best parametres and evaluate on the indeppendent test set 
    print(f"\n=== Retrain final {model_label} with params {best_params} ===")
    y_cv_cat = to_categorical(y_cv, num_classes=np.max(y_cv)+1)
    # data augmentation with validation split
    datagen = ImageDataGenerator(
        rotation_range=15, width_shift_range=0.08, height_shift_range=0.08,
        shear_range=0.05, zoom_range=0.08, horizontal_flip=True, fill_mode='nearest', validation_split=0.1
    )
    train_gen = datagen.flow(X_cv, y_cv_cat, batch_size=best_params['batch_size'], subset=None, shuffle=True, seed=SEED)
    # build model and compile
    model = build_fn(dropout=best_params['dropout'])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=best_params['lr']),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    # callbacks for training control
    callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
                 keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2)]
    # train/validation split for final training
    X_tr, X_val, y_tr_int, y_val_int = train_test_split(X_cv, y_cv, test_size=0.10, stratify=y_cv, random_state=SEED)
    y_tr = to_categorical(y_tr_int, num_classes=np.max(y_cv)+1)
    y_val = to_categorical(y_val_int, num_classes=np.max(y_cv)+1)
    train_gen = ImageDataGenerator(rotation_range=15, width_shift_range=0.08, height_shift_range=0.08,
                                   shear_range=0.05, zoom_range=0.08, horizontal_flip=True).flow(X_tr, y_tr, batch_size=best_params['batch_size'], shuffle=True, seed=SEED)
    val_gen = ImageDataGenerator().flow(X_val, y_val, batch_size=best_params['batch_size'], shuffle=False)
    # train final model
    history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS_FINAL, callbacks=callbacks, verbose=1)
    model.save(f"best_{model_label}.h5")
    print(f"Saved model best_{model_label}.h5")

    # Evaluate on test set
    y_test_int = y_test
    y_test_cat = to_categorical(y_test_int, num_classes=np.max(y_cv)+1)
    test_datagen = ImageDataGenerator()
    test_gen = test_datagen.flow(X_test, y_test_cat, batch_size=best_params['batch_size'], shuffle=False)
    preds = model.predict(test_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    # print classification report
    print("\n--- Test classification report ---")
    print(classification_report(y_test_int, y_pred, digits=4))
    # plot confusion matrix
    cm = confusion_matrix(y_test_int, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion matrix {model_label} on test")
    plt.show()

    # show some misclassified sample
    mis = np.where(y_pred != y_test_int)[0]
    print(f"Misclassified on test: {len(mis)} / {len(y_test_int)}")
    if len(mis)>0:
        plt.figure(figsize=(12,6))
        for i, idx in enumerate(mis[:8]):
            plt.subplot(2,4,i+1)
            plt.imshow(X_test[idx])
            plt.title(f"True:{class_names[y_test_int[idx]]}\nPred:{class_names[y_pred[idx]]}")
            plt.axis('off')
        plt.show()

    # plot training curves
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(history.history['loss'], label='train_loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.legend(); plt.title(f"{model_label} Loss")
    plt.subplot(1,2,2)
    plt.plot(history.history['accuracy'], label='train_acc')
    plt.plot(history.history['val_accuracy'], label='val_acc')
    plt.legend(); plt.title(f"{model_label} Accuracy")
    plt.show()

    # compute numerical performance metrics
    acc = accuracy_score(y_test_int, y_pred)
    prec = precision_score(y_test_int, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_test_int, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test_int, y_pred, average='weighted', zero_division=0)
    return {'model': model_label, 'test_acc': acc, 'test_prec': prec, 'test_rec': rec, 'test_f1': f1}

# ---------------- main ----------------
if __name__ == "__main__":
    # locate dataset for folders
    train_dir, test_dir = find_dirs(DEFAULT_PARENT)
    print("Train dir:", train_dir, "Test dir:", test_dir)

    # load all training images for CV
    print("Loading images for CV from:", train_dir)
    X_all, y_all, class_names, _ = load_images_from_folder(train_dir, img_size=IMG_SIZE)
    print("Loaded:", X_all.shape, "labels:", np.unique(y_all, return_counts=True))

    # If test_dir exists, load test set separately; else create hold-out split
    if test_dir:
        print("Loading provided test set from:", test_dir)
        X_test, y_test, _, _ = load_images_from_folder(test_dir, img_size=IMG_SIZE)
        X_cv, y_cv = X_all, y_all
    else:
        print("No test dir provided — creating hold-out test set (15%)")
        X_cv, X_test, y_cv, y_test = train_test_split(X_all, y_all, test_size=0.15, stratify=y_all, random_state=SEED)
        print("CV set:", X_cv.shape, "Test set:", X_test.shape)

    # Run grid+CV for all models
    results_summary = {}
    for label, build_fn in [('ModelA', build_model_A), ('ModelB', build_model_B), ('ModelC', build_model_C)]:
        res = run_grid_cv(X_cv, y_cv, build_fn, PARAM_GRID, label)
        best = res[0]   # best by mean_acc
        print(f"\nBEST for {label}: {best['params']} mean_acc={best['mean_acc']:.4f}")
        results_summary[label] = {'cv_results': res, 'best_params': best['params'], 'best_mean_acc': best['mean_acc'], 'best_std': best['std_acc']}

    # Retrain models with best params and evaluate on test set
    final_perfs = []
    for label, build_fn in [('ModelA', build_model_A), ('ModelB', build_model_B), ('ModelC', build_model_C)]:
        best_params = results_summary[label]['best_params']
        perf = retrain_and_evaluate(X_cv, y_cv, X_test, y_test, best_params, build_fn, label)
        final_perfs.append({'label': label, **perf, 'cv_mean_acc': results_summary[label]['best_mean_acc'], 'cv_std': results_summary[label]['best_std']})

    # print summary table
    import pandas as pd
    df_final = pd.DataFrame(final_perfs)
    print("\n=== SUMMARY TABLE ===")
    print(df_final)

    print("\nDONE.")
