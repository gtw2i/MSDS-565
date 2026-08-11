"""
msds565_helpers.py -- shared helper functions for the MSDS 565 course notebooks.

This module lives in the repository ROOT so every notebook can import it with a
relative path back to the root.  Notebooks live two folders below the root
(e.g. "Unit 2 - Image Data/Demos/"), so the standard import block does:

    import sys
    sys.path.append('../..')          # the repo root, where this file lives
    import msds565_helpers as helpers

Keeping reusable code here -- instead of copy-pasting it into every notebook --
means a fix or improvement made once is shared by all demos and projects.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_squared_error,
    mean_absolute_error,
    r2_score,
)


def _detect_task(model):
    """Guess whether a compiled Keras model does multiclass, binary, or regression.

    We look at two things:
      * the number of units in the output layer, and
      * the name of the loss the model was compiled with.
    Cross-entropy losses mean classification; anything else we treat as regression.
    """
    # Number of units in the final layer (e.g. 10 for MNIST, 1 for regression).
    try:
        n_outputs = model.output_shape[-1]
    except (AttributeError, TypeError):
        n_outputs = None

    # The loss may be a string ('mse'), a loss object, or a function -- get its name.
    loss = model.loss
    if isinstance(loss, str):
        loss_name = loss
    elif hasattr(loss, 'name'):
        loss_name = loss.name
    elif hasattr(loss, '__name__'):
        loss_name = loss.__name__
    else:
        loss_name = str(loss)
    loss_name = loss_name.lower()

    is_classification = ('crossentropy' in loss_name) or ('bce' in loss_name)

    if is_classification:
        # One output unit + sigmoid is binary; several output units is multiclass.
        return 'binary' if n_outputs == 1 else 'multiclass'
    if n_outputs is not None and n_outputs > 1:
        # A multi-unit output with a non-cross-entropy loss is still almost always
        # a (softmax) classifier in this course, so default to multiclass.
        return 'multiclass'
    return 'regression'


def train_and_evaluate(model, X_train, y_train, X_test, y_test,
                       epochs=10, batch_size=512, shuffle=True,
                       callbacks=None, verbose=1,
                       val_data=None, task='auto', class_names=None,
                       **fit_kwargs):
    """Fit a Keras model, then plot its learning curve and report performance.

    This single helper works for every kind of network built in this course --
    multiclass classification, binary classification, and regression.  It:

      1. trains the model with ``model.fit`` (early stopping etc. via *callbacks*),
      2. plots training/validation loss next to a confusion matrix
         (classification) or a predicted-vs-actual scatter (regression), and
      3. prints a classification report or regression metrics.

    Parameters
    ----------
    model : a compiled Keras model.
    X_train, y_train : training data.  To train on augmented / batched data,
        pass a data generator (e.g. ``datagen.flow(...)``) as *X_train* and set
        ``y_train=None`` -- the generator supplies the labels itself.
    X_test, y_test : test data, used both for validation and for evaluation.
    epochs, batch_size, shuffle, callbacks, verbose : forwarded to ``model.fit``.
    val_data : optional validation data to use instead of ``(X_test, y_test)``
        during training (handy when validating on augmented batches).
    task : 'auto' (default), 'multiclass', 'binary', or 'regression'.
    class_names : optional list of labels for the confusion-matrix axes / report.
    **fit_kwargs : any other keyword arguments passed straight to ``model.fit``.

    Returns
    -------
    (model, history) : the fitted model and its Keras ``History`` object.
    """
    # ----- 1. Train ------------------------------------------------------------
    validation_data = val_data if val_data is not None else (X_test, y_test)

    if y_train is None:
        # X_train is already a batched dataset / data generator, so we do NOT pass
        # y, batch_size, or shuffle -- the generator handles those itself.
        history = model.fit(
            X_train,
            epochs=epochs,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks,
            **fit_kwargs
        )
    else:
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=shuffle,
            validation_data=validation_data,
            verbose=verbose,
            callbacks=callbacks,
            **fit_kwargs
        )

    # ----- 2. Figure out what kind of problem this is --------------------------
    if task == 'auto':
        task = _detect_task(model)

    # ----- 3. Predict on the test set ------------------------------------------
    y_pred_raw = model.predict(X_test)
    y_true = np.asarray(y_test)

    if task == 'multiclass':
        # Softmax output -> pick the highest-probability class.
        y_pred = y_pred_raw.argmax(axis=1)
        # If the true labels are one-hot encoded, turn them back into integers.
        if y_true.ndim > 1 and y_true.shape[-1] > 1:
            y_true = y_true.argmax(axis=1)
    elif task == 'binary':
        # Single sigmoid output -> threshold the probability at 0.5.
        y_pred = (y_pred_raw.ravel() > 0.5).astype(int)
        y_true = y_true.ravel().astype(int)
    else:  # regression
        y_pred = y_pred_raw.ravel()
        y_true = y_true.ravel()

    # ----- 4. Plot: loss curve + a task-specific right-hand panel --------------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Left subplot: training and validation loss.
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].set_xlabel('Epochs')
    axes[0].set_ylabel('Loss')
    axes[0].legend()

    if task == 'regression':
        # Right subplot: predicted vs. actual, with a diagonal "perfect fit" line.
        axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)
        lo = min(y_true.min(), y_pred.min())
        hi = max(y_true.max(), y_pred.max())
        axes[1].plot([lo, hi], [lo, hi], 'r--', linewidth=1)
        axes[1].set_title('Predicted vs. Actual (Test Data)')
        axes[1].set_xlabel('Actual')
        axes[1].set_ylabel('Predicted')
    else:
        # Right subplot: confusion matrix as a seaborn heatmap (vmin=0 fixes the
        # color floor so an empty cell always reads as the "zero" color).
        cm = confusion_matrix(y_true, y_pred)
        tick_labels = class_names if class_names is not None else 'auto'
        sns.heatmap(cm, annot=True, fmt='d', ax=axes[1], vmin=0,
                    annot_kws={"size": 12}, cmap='nipy_spectral',
                    xticklabels=tick_labels, yticklabels=tick_labels)
        axes[1].set_title('Confusion Matrix (Test Data)')
        axes[1].set_xlabel('Predicted Label')
        axes[1].set_ylabel('True Label')

    plt.tight_layout()
    plt.show()

    # ----- 5. Print a text report ----------------------------------------------
    if task == 'regression':
        print("Test-set regression metrics")
        print(f"  MSE : {mean_squared_error(y_true, y_pred):.4f}")
        print(f"  MAE : {mean_absolute_error(y_true, y_pred):.4f}")
        print(f"  R^2 : {r2_score(y_true, y_pred):.4f}")
    elif class_names is not None:
        # Line up the report rows with the provided names (labels 0, 1, 2, ...).
        labels = np.arange(len(class_names))
        print(classification_report(y_true, y_pred, labels=labels,
                                    target_names=[str(c) for c in class_names]))
    else:
        print(classification_report(y_true, y_pred))

    return model, history


def visualize_layer_outputs(model, image, n):
    """Run a single image through the first ``n`` layers of a model and plot the
    output channels, so we can see what the convolutional filters respond to.

    Parameters
    ----------
    model : a Keras model.
    image : a single input image (its shape must match the model input).
    n : the layer index to read activations from (i.e. include the first n layers).
    """
    # Imported here (not at module top) so notebooks that don't use Keras don't
    # pay the cost of loading TensorFlow just to import this helper module.
    from tensorflow.keras.models import Model

    # Ensure image has batch dimension
    if image.ndim == 3:
        image = np.expand_dims(image, axis=0)

    # Create a model up to the nth layer
    truncated_model = Model(inputs=model.inputs, outputs=model.layers[n].output)

    # Get activations
    activations = truncated_model.predict(image)

    # Squeeze batch dimension if needed
    if activations.ndim == 4:
        activations = np.squeeze(activations, axis=0)  # shape: (H, W, C)

    n_channels = activations.shape[-1]

    # Set up subplot grid
    n_cols = 4
    n_rows = int(np.ceil((n_channels + 1) / n_cols))  # +1 for original image
    size = 4

    plt.figure(figsize=(size * n_cols, size * n_rows))

    # Plot original image
    plt.subplot(n_rows, n_cols, 1)
    if image.shape[-1] == 1:
        plt.imshow(image[0, ..., 0], cmap='gray')
    else:
        plt.imshow(image[0])
    plt.title("Original")

    # Plot each channel
    for i in range(n_channels):
        plt.subplot(n_rows, n_cols, i + 2)
        plt.imshow(activations[..., i], cmap='gray', interpolation=None)
        plt.title(f"Channel {i}")

    plt.tight_layout()
    plt.show()
