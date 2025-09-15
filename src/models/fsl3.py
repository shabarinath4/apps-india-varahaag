import logging
import random
import time
from typing import Dict, List, Optional, Tuple

import faiss
import numpy as np
import tensorflow as tf
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Conv1D,
    Dense,
    Flatten,
    Input,
    Lambda,
    MaxPooling1D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import numpy as np
import random
from typing import Dict, List, Tuple
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from src.models.inference import Infer
from src.models.triplet_mining import TripletMiner
from tensorflow.keras.layers import GlobalAveragePooling1D

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

class TemporalPyramidPooling(tf.keras.layers.Layer):
    def __init__(self, pyramid_levels=[1, 2], **kwargs):
        super().__init__(**kwargs)
        self.pyramid_levels = pyramid_levels

    def call(self, t):
        def pool_level(t, n):
            time_len = tf.shape(t)[1]
            boundaries = tf.cast(tf.linspace(0., tf.cast(time_len, tf.float32), n + 1), tf.int32)
            pooled_slices = []
            for i in range(n):
                start = boundaries[i]
                end = boundaries[i + 1]
                slice_ = t[:, start:end, ...]
                pooled = tf.reduce_max(slice_, axis=1)
                pooled_slices.append(pooled)
            return tf.concat(pooled_slices, axis=-1)

        pooled_levels = [pool_level(t, n) for n in self.pyramid_levels]
        return tf.concat(pooled_levels, axis=-1)

    def get_config(self):
        config = super().get_config()
        config.update({"pyramid_levels": self.pyramid_levels})
        return config

class Train:
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        COMB_PER_SAMPLE: int,
        training_params: Dict
    ):
        """
        Initializes the training setup.

        Parameters
        ----------
        x_train, y_train : Training data and labels
        x_test, y_test : Testing data and labels
        COMB_PER_SAMPLE : Number of combinations per sample for triplet creation
        training_params : Dictionary of training hyperparameters
        """
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = len(np.unique(self.y_train))
        self.COMB_PER_SAMPLE = COMB_PER_SAMPLE
        self.training_params = training_params

    def create_base_network(self, input_shape: Tuple[int, int]) -> Model:
        """
        Builds the embedding model using Conv1D + Temporal Pyramid Pooling.

        Parameters
        ----------
        input_shape : Tuple[int, int]
            Shape of the input data (time, features)

        Returns
        -------
        keras.Model : Compiled embedding model
        """
        inp = Input(shape=input_shape)
        x = Lambda(lambda t: tf.squeeze(t, axis=-1))(inp)           # Remove last channel if it's 1
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = TemporalPyramidPooling(pyramid_levels=[1, 2, 3, 4, 5])(x) # India
        x = Dense(64, activation='relu')(x)
        x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)    # L2 normalize embeddings
        print('Running temporal independent')
        return Model(inp, x)
    
    # def create_base_network(self, input_shape: Tuple[int, int]) -> Model:
    #     """
    #     Builds the embedding model using Conv1D only (no pooling).

    #     Parameters
    #     ----------
    #     input_shape : Tuple[int, int]
    #         Shape of the input data (time, features)

    #     Returns
    #     -------
    #     keras.Model : Embedding model
    #     """
    #     inp = Input(shape=input_shape)
    #     x = Lambda(lambda t: tf.squeeze(t, axis=-1))(inp)           # Remove last channel if it's 1
    #     x = Conv1D(16, 3, activation='relu', padding='same')(x)
    #     x = Conv1D(32, 3, activation='relu', padding='same')(x)
    #     x = Conv1D(64, 3, activation='relu', padding='same')(x)
    #     x = Dense(64, activation='relu')(x)                         # Dense applied per time step
    #     # Flatten all time steps to a single vector
    #     x = Flatten()(x)
    #     x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)    # L2 normalize embeddings
    #     print('Running temporal independent without pooling')
    #     return Model(inp, x)

    def triplet_model(
        self, 
        base_network: Model, 
        input_shape: Tuple[int, int]
    ) -> Model:
        """
        Builds a triplet model using the base network.

        Parameters
        ----------
        base_network : Embedding model
        input_shape : Shape of the input

        Returns
        -------
        keras.Model : Triplet network model
        """
        in_a = Input(shape=input_shape)
        in_p = Input(shape=input_shape)
        in_n = Input(shape=input_shape)
        emb_a = base_network(in_a)
        emb_p = base_network(in_p)
        emb_n = base_network(in_n)

        combined = Lambda(lambda x: K.concatenate(x, axis=-1))([emb_a, emb_p, emb_n])
        model = Model([in_a, in_p, in_n], combined)
        model.compile(loss=self.triplet_loss, optimizer=Adam(learning_rate=0.001))
        return model

    def classification_model(
        self, 
        base_network: Model, 
        input_shape: Tuple[int, int]
    ) -> Model:
        """
        Builds a classification model from the base embedding model.

        Parameters
        ----------
        base_network : Embedding model
        input_shape : Shape of input data

        Returns
        -------
        keras.Model : Classification model
        """
        input = Input(shape=input_shape)
        emb = base_network(input)
        out = Dense(self.num_classes, activation='softmax')(emb)
        model = Model(input, out)
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def triplet_loss(
        self, y_true: tf.Tensor, 
        y_pred: tf.Tensor, 
        alpha: float = 0.2
    ) -> tf.Tensor:
        total_length = y_pred.shape.as_list()[-1]
        anchor = y_pred[:, 0:int(total_length / 3)]
        positive = y_pred[:, int(total_length / 3):int(2 * total_length / 3)]
        negative = y_pred[:, int(2 * total_length / 3):]
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)
        loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
        return loss

    def train_model(
        self,
        loss_type: str = 'cross-entropy',
        triplet_strategy: str = 'standard',
        episodic: bool = False,
        episodes: int = 100,
        k_way: int = 5,
        n_shot: int = 5,
        n_query: int = 5,
        batch_size: int = 32,
        use_pretrained: bool = False,
        n_epochs: int = 10,
        pretrained_path: str = None,
        val_split: float = 0.2,
        patience: int = 5,
        domain_array: np.ndarray = None
    ):
        input_shape = (self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3])
        base_network = self.create_base_network(input_shape)

        if use_pretrained and pretrained_path:
            base_network.load_weights(pretrained_path)

        if loss_type == 'triplet':
            model = self.triplet_model(base_network, input_shape)
        else:
            model = self.classification_model(base_network, input_shape)

        x_train_main, x_val, y_train_main, y_val = train_test_split(
            self.x_train, self.y_train, test_size=val_split, random_state=42, stratify=self.y_train
        )

        miner = TripletMiner(comb_per_sample=self.COMB_PER_SAMPLE)

        # -------------------
        # Validation data
        # -------------------
        if loss_type == 'triplet':
            val_embeddings = base_network.predict(x_val)

            def val_triplet_generator():
                for i in range(0, len(x_val), batch_size):
                    x_batch = x_val[i:i+batch_size]
                    y_batch = y_val[i:i+batch_size]
                    emb_batch = val_embeddings[i:i+batch_size]

                    if triplet_strategy == 'standard':
                        triplets, _ = miner.standard_triplets(x_batch, emb_batch, y_batch)
                    elif triplet_strategy == 'batch-hard':
                        triplets, _ = miner.batch_hard_triplets(x_batch, emb_batch, y_batch, samples_per_class=k_way)
                    else:
                        triplets, _ = miner.domain_aware_triplets(x_batch, emb_batch, y_batch, domain_array, len(np.unique(self.y_train)))

                    if len(triplets) == 0:
                        continue

                    a_idx, p_idx, n_idx = triplets[:,0], triplets[:,1], triplets[:,2]
                    a = x_batch[a_idx][..., np.newaxis]
                    p = x_batch[p_idx][..., np.newaxis]
                    n = x_batch[n_idx][..., np.newaxis]

                    # Yield as tuple of three tensors, not a single list
                    yield (a, p, n), np.zeros(len(a))

            val_data = tf.data.Dataset.from_generator(
                val_triplet_generator,
                output_signature=(
                    (
                        tf.TensorSpec(shape=(None, *x_val.shape[1:], 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, *x_val.shape[1:], 1), dtype=tf.float32),
                        tf.TensorSpec(shape=(None, *x_val.shape[1:], 1), dtype=tf.float32)
                    ),
                    tf.TensorSpec(shape=(None,), dtype=tf.float32)
                )
            ).prefetch(2)

        else:
            x_val_cat = x_val[..., np.newaxis]
            y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=k_way)
            val_data = (x_val_cat, y_val_cat)

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # -------------------
        # Training loop
        # -------------------
        if loss_type == 'triplet':
            for epoch in range(n_epochs):
                print(f"\n=== Epoch {epoch+1}/{n_epochs} ===")
                embeddings = base_network.predict(x_train_main)

                for i in range(0, len(x_train_main), batch_size):
                    x_batch = x_train_main[i:i+batch_size]
                    y_batch = y_train_main[i:i+batch_size]
                    emb_batch = embeddings[i:i+batch_size]

                    if triplet_strategy == 'standard':
                        triplets, _ = miner.standard_triplets(x_batch, emb_batch, y_batch)
                    elif triplet_strategy == 'batch-hard':
                        triplets, _ = miner.batch_hard_triplets(x_batch, emb_batch, y_batch, samples_per_class=k_way)
                    else:
                        triplets, _ = miner.domain_aware_triplets(x_batch, emb_batch, y_batch, domain_array, len(np.unique(self.y_train)))

                    if len(triplets) == 0:
                        continue

                    a_idx, p_idx, n_idx = triplets[:,0], triplets[:,1], triplets[:,2]
                    a = x_batch[a_idx][..., np.newaxis]
                    p = x_batch[p_idx][..., np.newaxis]
                    n = x_batch[n_idx][..., np.newaxis]

                    model.fit(
                        [a, p, n],
                        np.zeros(len(a)),
                        batch_size=len(a),
                        epochs=1,
                        verbose=1,
                        validation_data=val_data,
                        callbacks=[early_stopping]
                    )

            train_embeddings = base_network.predict(self.x_train)
            train_index, train_labels = Infer().make_faiss_index(train_embeddings, self.y_train)

        else:
            x_train_input = x_train_main[..., np.newaxis]
            y_train_input = tf.keras.utils.to_categorical(y_train_main, num_classes=k_way)
            model.fit(
                x_train_input, y_train_input,
                batch_size=batch_size,
                epochs=n_epochs,
                verbose=1,
                validation_data=val_data,
                callbacks=[early_stopping]
            )

        return (
            base_network if loss_type == 'triplet' else model,
            None,
            None,
            train_embeddings if loss_type == 'triplet' else None,
            self.training_params,
            self.y_train,
            train_index if loss_type == 'triplet' else None,
            train_labels if loss_type == 'triplet' else None
        )


    def finetune(
        self,
        base_network: tf.keras.Model,
        support_x: np.ndarray,
        support_y: np.ndarray,
        loss_type: str,
        n_epochs: int,
        batch_size: int,
        freeze_base: bool,
        val_split: float = 0.2,
        patience: int = 5,
        triplet_strategy: str = 'standard',  # 'standard' or 'domain-aware'
        domain_array: np.ndarray = None
    ):

        input_shape = support_x.shape[1:]
        
        # Freeze or unfreeze base layers
        for layer in base_network.layers:
            layer.trainable = not freeze_base

        x_train_main, x_val, y_train_main, y_val = train_test_split(
            support_x, support_y, test_size=val_split, stratify=support_y, random_state=42
        )

        miner = TripletMiner(comb_per_sample=self.COMB_PER_SAMPLE)
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        if loss_type == 'triplet':
            if freeze_base:
                x = base_network.output
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                embedding_output = tf.keras.layers.Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
                base_network = tf.keras.Model(inputs=base_network.input, outputs=embedding_output)

            model = self.triplet_model(base_network, input_shape)
            num_classes = len(np.unique(y_train_main))

            # Prepare validation triplets
            val_embeddings = base_network.predict(x_val)
            if triplet_strategy == 'standard':
                val_triplets, _ = miner.standard_triplets(x_val, val_embeddings, y_val, num_classes)
            else:
                val_triplets, _ = miner.domain_aware_triplets(x_val, val_embeddings, y_val, domain_array, num_classes)

            a_val, p_val, n_val = val_triplets[:, 0], val_triplets[:, 1], val_triplets[:, 2]
            a_val, p_val, n_val = a_val[..., np.newaxis], p_val[..., np.newaxis], n_val[..., np.newaxis]
            val_data = ([a_val, p_val, n_val], np.zeros(len(a_val)))

            # Training loop
            for epoch in range(n_epochs):
                embeddings = base_network.predict(x_train_main)
                if triplet_strategy == 'standard':
                    triplets, _ = miner.standard_triplets(x_train_main, embeddings, y_train_main, num_classes)
                else:
                    triplets, _ = miner.domain_aware_triplets(x_train_main, embeddings, y_train_main, domain_array, num_classes)

                a, p, n = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                a, p, n = a[..., np.newaxis], p[..., np.newaxis], n[..., np.newaxis]

                model.fit(
                    [a, p, n], np.zeros(len(a)),
                    batch_size=batch_size,
                    epochs=1,
                    verbose=1,
                    validation_data=val_data,
                    callbacks=[early_stopping]
                )

            train_embeddings = base_network.predict(x_train_main)
            train_index, train_labels = Infer().make_faiss_index(train_embeddings, y_train_main)

        elif loss_type == 'classification':
            if freeze_base:
                x = base_network.output
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                output = tf.keras.layers.Dense(len(np.unique(y_train_main)), activation='softmax')(x)
                base_network = tf.keras.Model(inputs=base_network.input, outputs=output)

            model = base_network

            y_train_cat = to_categorical(y_train_main, num_classes=len(np.unique(y_train_main)))
            y_val_cat = to_categorical(y_val, num_classes=len(np.unique(y_val)))

            x_train_main_cat = x_train_main[..., np.newaxis]
            x_val_cat = x_val[..., np.newaxis]

            model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

            model.fit(
                x_train_main_cat, y_train_cat,
                batch_size=batch_size,
                epochs=n_epochs,
                verbose=1,
                validation_data=(x_val_cat, y_val_cat),
                callbacks=[early_stopping]
            )

            train_embeddings, train_index, train_labels = None, None, None

        else:
            raise ValueError("Invalid loss_type. Use 'triplet' or 'classification'.")

        return (
            base_network,
            None,
            None,
            x_train_main,
            train_embeddings,
            self.training_params,
            y_train_main,
            train_index,
            train_labels
        )

    

