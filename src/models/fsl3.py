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

    # def create_base_network(self, input_shape: Tuple[int, int]) -> Model:
    #     """
    #     Builds the embedding model using Conv1D + Temporal Pyramid Pooling.

    #     Parameters
    #     ----------
    #     input_shape : Tuple[int, int]
    #         Shape of the input data (time, features)

    #     Returns
    #     -------
    #     keras.Model : Compiled embedding model
    #     """
    #     inp = Input(shape=input_shape)
    #     x = Lambda(lambda t: tf.squeeze(t, axis=-1))(inp)           # Remove last channel if it's 1
    #     x = Conv1D(16, 3, activation='relu', padding='same')(x)
    #     x = Conv1D(32, 3, activation='relu', padding='same')(x)
    #     x = Conv1D(64, 3, activation='relu', padding='same')(x)
    #     x = TemporalPyramidPooling(pyramid_levels=[1, 3, 5, 7])(x)
    #     x = Dense(32, activation='relu')(x)
    #     x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)    # L2 normalize embeddings
    #     print('Running temporal independent')
    #     return Model(inp, x)

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
            input = Input(shape=input_shape)
            x = Conv1D(16, (3), activation='relu', padding='same')(input)  
            x = MaxPooling2D(pool_size=(1, 1))(x)
            x = Conv1D(32, (3), activation='relu', padding='same')(x)  
            x = MaxPooling2D(pool_size=(1, 1))(x)
            x = Conv1D(64, (3), activation='relu', padding='same')(x)  
            x = MaxPooling2D(pool_size=(1, 1))(x)
            x = Flatten()(x)  
            x = Dense(32, activation='sigmoid')(x)  
            return Model(input, x)

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
        """
        Computes the triplet loss.

        Parameters
        ----------
        y_true : Tensor of true labels (unused)
        y_pred : Concatenated embeddings of anchor, positive, negative
        alpha : Margin

        Returns
        -------
        tf.Tensor : Triplet loss value
        """
        total_length = y_pred.shape.as_list()[-1]
        anchor = y_pred[:, 0:int(total_length / 3)]
        positive = y_pred[:, int(total_length / 3):int(2 * total_length / 3)]
        negative = y_pred[:, int(2 * total_length / 3):]
        pos_dist = K.sum(K.square(anchor - positive), axis=1)
        neg_dist = K.sum(K.square(anchor - negative), axis=1)
        loss = K.maximum(pos_dist - neg_dist + alpha, 0.0)
        return loss

    def stratified_support_query_split(
        self,
        x,
        y,
        support_ratio=0.7,
        seed=None,
        mode="original",  # or "cluster"
        n_clusters=5
    ):
        """
        Splits data into support/query sets, either using standard stratified sampling
        or clustering within each class and sampling from all clusters.

        Parameters
        ----------
        x : np.ndarray
            Feature array.
        y : np.ndarray
            Label array.
        support_ratio : float
            Proportion of data to use for support set.
        seed : int or None
            Random seed.
        mode : str
            "original" for normal stratified split,
            "cluster" for per-class cluster-aware stratified split.
        n_clusters : int
            Number of clusters to try within each class for cluster mode.

        Returns
        -------
        support_x, support_y, query_x, query_y : np.ndarrays
        """
        support_x, support_y = [], []
        query_x, query_y = [], []
        unique_classes = np.unique(y)

        for cls in unique_classes:
            cls_indices = np.where(y == cls)[0]
            cls_x = x[cls_indices]
            cls_y = y[cls_indices]

            if mode == "original" or len(cls_indices) < 2:
                # Fallback to basic stratified split if too few samples
                x_supp, x_query, y_supp, y_query = train_test_split(
                    cls_x, cls_y,
                    train_size=support_ratio,
                    stratify=cls_y if len(np.unique(cls_y)) > 1 else None,
                    random_state=seed
                )
            elif mode == "cluster":
                # Try KMeans clustering inside each class
                clusters = min(n_clusters, len(cls_indices))
                if clusters < 2:
                    x_supp, x_query, y_supp, y_query = train_test_split(
                        cls_x, cls_y,
                        train_size=support_ratio,
                        random_state=seed
                    )
                else:
                    band_index = -3  
                    band_data = cls_x[:, :, band_index, 0]  
                    flattened_band = band_data.reshape(cls_x.shape[0], -1)

                    kmeans = KMeans(n_clusters=clusters, random_state=seed)
                    cluster_labels = kmeans.fit_predict(flattened_band)
                    x_supp_parts, y_supp_parts = [], []
                    x_query_parts, y_query_parts = [], []

                    for cluster_id in range(clusters):
                        cluster_mask = cluster_labels == cluster_id
                        cluster_x = cls_x[cluster_mask]
                        cluster_y = cls_y[cluster_mask]

                        if len(cluster_y) < 2:
                            x_supp_parts.append(cluster_x)
                            y_supp_parts.append(cluster_y)
                        else:
                            x_s, x_q, y_s, y_q = train_test_split(
                                cluster_x, cluster_y,
                                train_size=support_ratio,
                                random_state=seed
                            )
                            x_supp_parts.append(x_s)
                            y_supp_parts.append(y_s)
                            x_query_parts.append(x_q)
                            y_query_parts.append(y_q)

                    x_supp = np.concatenate(x_supp_parts)
                    y_supp = np.concatenate(y_supp_parts)
                    x_query = np.concatenate(x_query_parts) if x_query_parts else np.array([])
                    y_query = np.concatenate(y_query_parts) if y_query_parts else np.array([])

            else:
                raise ValueError(f"Invalid mode '{mode}'. Use 'original' or 'cluster'.")

            support_x.append(x_supp)
            support_y.append(y_supp)
            query_x.append(x_query)
            query_y.append(y_query)

        support_x = np.concatenate(support_x)
        support_y = np.concatenate(support_y)
        query_x = np.concatenate(query_x) if query_x else np.array([])
        query_y = np.concatenate(query_y) if query_y else np.array([])

        return support_x, support_y, query_x, query_y


    def sample_episode(
        self,
        x: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        n_support: int,
        n_query: int,
        seed: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Samples an episode for few-shot learning.

        Returns
        -------
        support_x, support_y, query_x, query_y
        """
        np.random.seed(seed)
        selected_classes = np.random.choice(np.unique(y), num_classes, replace=False)
        support_x, support_y, query_x, query_y = [], [], [], []

        for idx, cls in enumerate(selected_classes):
            cls_indices = np.where(y == cls)[0]
            chosen = np.random.choice(cls_indices, n_support + n_query, replace=True)
            support_x.append(x[chosen[:n_support]])
            query_x.append(x[chosen[n_support:]])
            support_y += [idx] * n_support
            query_y += [idx] * n_query

        return (
            np.concatenate(support_x), np.array(support_y),
            np.concatenate(query_x), np.array(query_y)
        )

    def create_triplets(
        self,
        x: np.ndarray,
        embeddings: np.ndarray,
        y: np.ndarray,
        num_classes: int,
        comb_per_sample: int,
        seed: int = 42,
        alpha: float = 0.3,
        triplet_logic_ratio: Dict[str, float] = {'semi-hard': 0.7, 'hard': 0.15, 'easy': 0.15}
    ) -> Tuple[np.ndarray, List[int]]:
        np.random.seed(seed)
        random.seed(seed)
        triplets = []
        triplet_classes = []

        indices = [np.where(y == i)[0] for i in range(num_classes)]

        for d in range(num_classes):
            class_indices = indices[d]
            if len(class_indices) < 2:
                continue

            class_embeds = embeddings[class_indices]
            class_x = x[class_indices]
            n_samples = len(class_embeds)
            class_embeds_rs = class_embeds.reshape(n_samples, -1)

            for i in range(n_samples):
                anchor_embed = class_embeds_rs[i].reshape(1, -1)
                anchor_x = class_x[i]
                positive_candidates = np.delete(np.arange(n_samples), i)
                if len(positive_candidates) == 0:
                    continue

                for _ in range(comb_per_sample):
                    pos_idx = random.choice(positive_candidates)
                    positive_embed = class_embeds_rs[pos_idx].reshape(1, -1)
                    positive_x = class_x[pos_idx]
                    anchor_positive_dist = np.linalg.norm(anchor_embed - positive_embed)

                    other_classes = [j for j in range(num_classes) if j != d and len(indices[j]) > 0]
                    if not other_classes:
                        continue
                    neg_class = random.choice(other_classes)
                    neg_embeds = embeddings[indices[neg_class]]
                    neg_x = x[indices[neg_class]]
                    neg_embeds_rs = neg_embeds.reshape(len(neg_embeds), -1)
                    neg_dists = pairwise_distances(anchor_embed, neg_embeds_rs).flatten()

                    hard_mask = neg_dists < anchor_positive_dist
                    semi_hard_mask = (neg_dists >= anchor_positive_dist) & (neg_dists < anchor_positive_dist + alpha)
                    easy_mask = neg_dists >= anchor_positive_dist + alpha

                    hard = np.where(hard_mask)[0]
                    semi_hard = np.where(semi_hard_mask)[0]
                    easy = np.where(easy_mask)[0]

                    selected_neg = None
                    r = random.random()
                    if r < triplet_logic_ratio['hard'] and len(hard) > 0:
                        selected_neg = neg_x[random.choice(hard)]
                    elif r < triplet_logic_ratio['hard'] + triplet_logic_ratio['semi-hard'] and len(semi_hard) > 0:
                        selected_neg = neg_x[random.choice(semi_hard)]
                    elif len(easy) > 0:
                        selected_neg = neg_x[random.choice(easy)]
                    else:
                        selected_neg = neg_x[np.argmax(neg_dists)]

                    triplets.append([anchor_x, positive_x, selected_neg])
                    triplet_classes.append(d)

        indices = np.arange(len(triplets))
        np.random.shuffle(indices)
        return np.array(triplets)[indices], [triplet_classes[i] for i in indices]


    def train_model(
        self,
        loss_type: str = 'cross-entropy',
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
        patience: int = 5
    ) -> Model:

        input_shape = (self.x_train.shape[1], self.x_train.shape[2], self.x_train.shape[3])
        base_network = self.create_base_network(input_shape)

        if use_pretrained and pretrained_path:
            base_network.load_weights(pretrained_path)

        if loss_type == 'triplet':
            model = self.triplet_model(base_network, input_shape)
        else:
            model = self.classification_model(base_network, input_shape)

        x_train_main, x_val, y_train_main, y_val = train_test_split(
            self.x_train, self.y_train, test_size=val_split, random_state=42, stratify=self.y_train)

        if loss_type == 'triplet':
            val_embeddings = base_network.predict(x_val)
            val_triplets, _ = self.create_triplets(x_val, val_embeddings, y_val, len(np.unique(self.y_train)), self.COMB_PER_SAMPLE)
            a_val, p_val, n_val = val_triplets[:, 0], val_triplets[:, 1], val_triplets[:, 2]
            a_val, p_val, n_val = a_val[..., np.newaxis], p_val[..., np.newaxis], n_val[..., np.newaxis]
            val_data = ([a_val, p_val, n_val], np.zeros(len(a_val)))
        else:
            x_val_cat = x_val[..., np.newaxis]
            y_val_cat = tf.keras.utils.to_categorical(y_val, num_classes=k_way)
            val_data = (x_val_cat, y_val_cat)

        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        train_losses = []
        val_losses = []
        if not use_pretrained:
            if episodic:
                log.info("Episodic training")
                for e in range(episodes):
                    support_x, support_y, query_x, query_y = self.sample_episode(
                        x_train_main, y_train_main, k_way, n_shot, n_query)
                    all_x = np.concatenate([support_x, query_x])
                    all_y = np.concatenate([support_y, query_y])

                    if loss_type == 'triplet':
                        log.info(f"Episode {e}")
                        for epoch in range(n_epochs):
                            embeddings = base_network.predict(all_x)
                            triplets, _ = self.create_triplets(all_x, embeddings, all_y, k_way, self.COMB_PER_SAMPLE)
                            a, p, n = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                            a, p, n = a[..., np.newaxis], p[..., np.newaxis], n[..., np.newaxis]

                            history = model.fit(
                                [a, p, n], np.zeros(len(a)),
                                batch_size=batch_size,
                                epochs=epoch,
                                verbose=1,
                                validation_data=val_data,
                                callbacks=[early_stopping]
                            )

                        train_embeddings = base_network.predict(self.x_train)
                        train_index, train_labels = Infer().make_faiss_index(train_embeddings, self.y_train)

                    else:
                        x_cat = all_x[..., np.newaxis]
                        y_cat = tf.keras.utils.to_categorical(all_y, num_classes=k_way)

                        history = model.fit(
                            x_cat, y_cat,
                            batch_size=batch_size,
                            epochs=n_epochs,
                            verbose=1,
                            validation_data=val_data
                        )
            else:
                log.info("Non-episodic training on full dataset")

                if loss_type == 'triplet':
                    for epoch in range(n_epochs):
                        embeddings = base_network.predict(x_train_main)
                        triplets, _ = self.create_triplets(x_train_main, embeddings, y_train_main, len(np.unique(self.y_train)), self.COMB_PER_SAMPLE)
                        a, p, n = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                        a, p, n = a[..., np.newaxis], p[..., np.newaxis], n[..., np.newaxis]

                        history = model.fit(
                            [a, p, n], np.zeros(len(a)),
                            batch_size=batch_size,
                            epochs=epoch,
                            verbose=1,
                            validation_data=val_data,
                            callbacks=[early_stopping]
                        )

                    train_embeddings = base_network.predict(self.x_train)
                    train_index, train_labels = Infer().make_faiss_index(train_embeddings, self.y_train)

                else:
                    x_train_input = x_train_main[..., np.newaxis]
                    y_train_input = tf.keras.utils.to_categorical(y_train_main, num_classes=k_way)

                    history = model.fit(
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
        patience: int = 5
    ):

        input_shape = support_x.shape[1:]
        
        for layer in base_network.layers:
            layer.trainable = not freeze_base

        x_train_main, x_val, y_train_main, y_val = train_test_split(
            support_x, support_y, test_size=val_split, stratify=support_y, random_state=42)

        if loss_type == 'triplet':
            if freeze_base:
                x = base_network.output
                x = tf.keras.layers.Dense(128, activation='relu')(x)
                embedding_output = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
                base_network = tf.keras.Model(inputs=base_network.input, outputs=embedding_output)

            model = self.triplet_model(base_network, input_shape)
            num_classes = len(np.unique(y_train_main))

            val_embeddings = base_network.predict(x_val)
            val_triplets, _ = self.create_triplets(x_val, val_embeddings, y_val, num_classes, self.COMB_PER_SAMPLE)
            a_val, p_val, n_val = val_triplets[:, 0], val_triplets[:, 1], val_triplets[:, 2]
            a_val, p_val, n_val = a_val[..., np.newaxis], p_val[..., np.newaxis], n_val[..., np.newaxis]
            val_data = ([a_val, p_val, n_val], np.zeros(len(a_val)))

            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

            for epoch in range(n_epochs):
                embeddings = base_network.predict(x_train_main)
                triplets, _ = self.create_triplets(x_train_main, embeddings, y_train_main, num_classes, self.COMB_PER_SAMPLE)
                a, p, n = triplets[:, 0], triplets[:, 1], triplets[:, 2]
                a, p, n = a[..., np.newaxis], p[..., np.newaxis], n[..., np.newaxis]

                model.fit(
                    [a, p, n], np.zeros(len(a)),
                    batch_size=batch_size,
                    epochs=epoch,
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

            early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

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

    def few_shot_predict(
        model,
        x_data: np.ndarray,
        y_data: np.ndarray,
        n_query: int = 15,
        support_sizes: list = [1, 5, 10, 20, 100, 'all'],
        num_classes: int = None,
        episodes_per_setting: int = 10,
        verbose: bool = True
    ):
        from sklearn.preprocessing import LabelEncoder

        le = LabelEncoder()
        y_data = le.fit_transform(y_data)
        all_classes = np.unique(y_data)
        num_classes = num_classes or len(all_classes)

        results = {}
        fs_embeddings = model.predict(x_data, verbose=0)
        fs_index, fs_labels = Infer().make_faiss_index(fs_embeddings, y_data)

        for n_support in support_sizes:
            accs = []

            for _ in range(episodes_per_setting):
                chosen_classes = np.random.choice(all_classes, size=num_classes, replace=False)

                support_x, support_y, query_x, query_y = [], [], [], []

                for cls in chosen_classes:
                    cls_idx = np.where(y_data == cls)[0]
                    if len(cls_idx) <= n_query:
                        continue

                    np.random.shuffle(cls_idx)

                    if n_support == 'all':
                        support_idx = cls_idx[:-n_query]
                    else:
                        max_possible_support = len(cls_idx) - n_query
                        if max_possible_support <= 0:
                            continue
                        use_n = min(n_support, max_possible_support)
                        support_idx = cls_idx[:use_n]

                    query_idx = cls_idx[-n_query:]

                    if len(support_idx) == 0:
                        continue

                    support_x.append(x_data[support_idx])
                    support_y.extend([cls] * len(support_idx))
                    query_x.append(x_data[query_idx])
                    query_y.extend([cls] * len(query_idx))

                if len(support_y) == 0 or len(query_y) == 0:
                    continue

                support_x = np.concatenate(support_x)
                query_x = np.concatenate(query_x)

                support_emb = model.predict(support_x, verbose=0)
                query_emb = model.predict(query_x, verbose=0)

                prototypes = []
                unique_support_classes = np.unique(support_y)
                for cls in unique_support_classes:
                    cls_emb = support_emb[np.array(support_y) == cls]
                    prototypes.append(cls_emb.mean(axis=0))
                prototypes = np.stack(prototypes)

                dists = np.linalg.norm(query_emb[:, None, :] - prototypes[None, :, :], axis=-1)
                preds = unique_support_classes[np.argmin(dists, axis=1)]

                acc = np.mean(preds == np.array(query_y))
                accs.append(acc)

            avg_acc = np.mean(accs) if accs else 0.0
            results[n_support] = avg_acc

            if verbose:
                print(f"{n_support}-shot accuracy: {avg_acc:.4f}")

        return (
            model,
            None,
            None,
            x_data,
            fs_embeddings,
            {"few_shot_results": results},
            y_data,
            fs_index,
            fs_labels
        )

class Infer:
    def __init__(self):
        pass

    def make_faiss_index(
        self, 
        embeddings: np.ndarray, 
        labels: np.ndarray
    ):
        """
        Make a faiss index for faster similarity search

        Parameters
        ----------
        embeddings : np.ndarray
            embedding vector
        labels : np.ndarray
            label array
        
        Returns
        -------
        index : faiss.IndexFlatL2
        labels : np.ndarray
        """
        d = embeddings.shape[1]
        index = faiss.IndexFlatL2(d)
        index.add(embeddings.astype('float32'))
        return index, labels

    def predict_using_faiss(
        self, 
        index: faiss.IndexFlatL2, 
        labels: np.ndarray, 
        query_embeddings: np.ndarray, 
        k: int = 3
    ):
        """
        Make a prediction by searching the query in the index database

        Parameters
        ----------
        index : faiss.IndexFlatL2
            index to search in 
        labels : np.ndarray
            label array
        query_embeddings : np.ndarray
            embedding vector
        k : int
            number of nearest neighbors to search for a match
        
        Returns
        -------
        predicted_labels : np.ndarray
        distances : np.ndarray
        """
        t0 = time.time() 
        distances, indices = index.search(query_embeddings.astype('float32'), k)
        labels = np.array(labels)
        t1 = time.time() 
        print(f"Search took {t1 - t0:.2f} seconds")
        
        t0 = time.time() 
        if k > 1:
            predicted_labels = labels[indices]
            predicted_labels = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=1, arr=predicted_labels)
        else:
            indices = np.clip(indices.flatten(), 0, len(labels) - 1)
            predicted_labels = labels[indices]
        t1 = time.time() 
        print(f"Prediction took {t1 - t0:.2f} seconds")

        return predicted_labels, distances
    
    def infer(
        self, 
        base_network: tf.keras.Model, 
        support_emb: np.ndarray, 
        support_y: np.ndarray, 
        support_index: faiss.IndexFlatL2 , 
        x_infer: np.ndarray
    ):
        """
        Make a prediction by searching the query in the index database

        Parameters
        ----------
        base_network : tf.keras.Model
            The base embedding model to predict using
        support_emb : np.ndarray
            support embeddings
        support_y : np.ndarray
            support labels
        support_index : faiss.IndexFlatL2
            support index database
        x_infer : np.ndarray
            query array for inference

        Returns
        -------
        predicted_labels : np.ndarray
        distances : np.ndarray
        """
        query_emb = base_network.predict(x_infer, verbose=0)
        if support_index == None:
            support_index, support_y = Infer().make_faiss_index(support_emb, support_y)
        predicted_labels, distances = self.predict_using_faiss(support_index, support_y, query_emb, k=3)
        return predicted_labels, distances

class Evaluate:
    def __init__(self):
        pass

    def predict_class(
        self, 
        train_emb: np.ndarray, 
        y_train: np.ndarray, 
        test_emb: np.ndarray, 
        y_test: np.ndarray, 
        train_index: faiss.IndexFlatL2
    ):
        """
        Predict the class of a query

        Parameters
        ----------
        test_emb : np.ndarray
            test embeddings
        y_test : np.ndarray
            test labels
        train_emb : np.ndarray
            train embeddings
        y_train : np.ndarray
            train labels
        train_index : faiss.IndexFlatL2
            index of train data

        Returns
        -------
        predicted_labels : np.ndarray
        distances : np.ndarray
        """
        if train_index == None:
            train_index, y_train = Infer().make_faiss_index(train_emb, y_train)
        y_pred, distances = Infer().predict_using_faiss(train_index, y_train, test_emb)
        accuracy = np.mean(y_pred == y_test)
        return accuracy, y_pred, y_test