"""TensorFlow/Keras model helpers for queue text classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import Input, Model, layers, regularizers

from ticketpilot.config import (
    TENSORFLOW_CONV_FILTERS,
    TENSORFLOW_EMBEDDING_DIM,
    TENSORFLOW_MAX_TOKENS,
    TENSORFLOW_RANDOM_SEED,
    TENSORFLOW_SEQUENCE_LENGTH,
)


@dataclass(frozen=True)
class QueueLabelEncoder:
    """Small deterministic label encoder for queue names."""

    classes: tuple[str, ...]

    @classmethod
    def fit(cls, labels: pd.Series) -> QueueLabelEncoder:
        """Create an encoder from observed labels."""
        return cls(classes=tuple(sorted(str(label) for label in labels.unique())))

    def transform(self, labels: pd.Series) -> np.ndarray:
        """Encode string labels as integer class IDs."""
        class_to_index = {label: index for index, label in enumerate(self.classes)}
        return np.asarray(
            [class_to_index[str(label)] for label in labels], dtype="int32"
        )

    def inverse_transform(self, encoded: np.ndarray) -> np.ndarray:
        """Decode integer class IDs to queue names."""
        return np.asarray([self.classes[int(index)] for index in encoded])


def set_tensorflow_seed(seed: int = TENSORFLOW_RANDOM_SEED) -> None:
    """Set practical TensorFlow, NumPy, and Keras random seeds."""
    tf.keras.utils.set_random_seed(seed)


def build_text_vectorizer(
    *,
    max_tokens: int = TENSORFLOW_MAX_TOKENS,
    sequence_length: int = TENSORFLOW_SEQUENCE_LENGTH,
) -> layers.TextVectorization:
    """Build a TextVectorization layer for ticket text."""
    return layers.TextVectorization(
        max_tokens=max_tokens,
        output_mode="int",
        output_sequence_length=sequence_length,
        standardize="lower_and_strip_punctuation",
        split="whitespace",
    )


def adapt_text_vectorizer(
    vectorizer: layers.TextVectorization,
    train_text: pd.Series,
) -> None:
    """Adapt the vectorizer on training text only."""
    vectorizer.adapt(train_text.astype(str).to_numpy())


def build_conv1d_queue_model(
    *,
    vectorizer: layers.TextVectorization,
    num_classes: int,
    max_tokens: int = TENSORFLOW_MAX_TOKENS,
    embedding_dim: int = TENSORFLOW_EMBEDDING_DIM,
    conv_filters: int = TENSORFLOW_CONV_FILTERS,
) -> Model:
    """Build a compact TextVectorization + Embedding + Conv1D classifier."""
    inputs = Input(shape=(), dtype=tf.string, name="classifier_text")
    x = vectorizer(inputs)
    x = layers.Embedding(
        input_dim=max_tokens,
        output_dim=embedding_dim,
        embeddings_regularizer=regularizers.l2(1e-6),
        name="token_embedding",
    )(x)
    x = layers.SpatialDropout1D(0.20, name="embedding_dropout")(x)
    x = layers.Conv1D(
        filters=conv_filters,
        kernel_size=5,
        activation="relu",
        padding="same",
        kernel_regularizer=regularizers.l2(1e-5),
        name="local_text_patterns",
    )(x)
    x = layers.GlobalMaxPooling1D(name="global_max_pooling")(x)
    x = layers.Dropout(0.40, name="classifier_dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="queue")(x)
    model = Model(inputs=inputs, outputs=outputs, name="ticketpilot_conv1d_queue")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def compute_class_weights(
    encoded_labels: np.ndarray, *, num_classes: int
) -> dict[int, float]:
    """Return inverse-frequency class weights for sparse class IDs."""
    counts = np.bincount(encoded_labels.astype("int32"), minlength=num_classes)
    total = int(counts.sum())
    weights: dict[int, float] = {}
    for class_index, count in enumerate(counts):
        weights[class_index] = float(total / (num_classes * count)) if count else 0.0
    return weights
