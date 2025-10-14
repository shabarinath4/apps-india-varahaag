import logging
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

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


class TrainSupCon:
    def __init__(self, x_train, y_train, x_test, y_test, training_params):
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.num_classes = len(np.unique(y_train))
        self.training_params = training_params

    def create_base_network(self, input_shape):
        inp = Input(shape=input_shape)
        x = Lambda(lambda t: tf.squeeze(t, axis=-1))(inp)
        x = Conv1D(16, 3, activation='relu', padding='same')(x)
        x = Conv1D(32, 3, activation='relu', padding='same')(x)
        x = Conv1D(64, 3, activation='relu', padding='same')(x)
        x = TemporalPyramidPooling(pyramid_levels=[1, 2, 3, 4, 5])(x)
        x = Dense(64, activation='relu')(x)
        x = Lambda(lambda t: tf.math.l2_normalize(t, axis=1))(x)
        return Model(inp, x)

    def supcon_loss(self, features, labels, temperature=0.1):
        labels = tf.cast(labels, tf.int32)
        features = tf.math.l2_normalize(features, axis=1)
        batch_size = tf.shape(features)[0]
        mask = tf.equal(tf.expand_dims(labels, 1), tf.expand_dims(labels, 0))
        mask = tf.cast(mask, tf.float32)
        logits = tf.matmul(features, tf.transpose(features)) / temperature
        logits_max = tf.reduce_max(logits, axis=1, keepdims=True)
        logits = logits - logits_max
        exp_logits = tf.exp(logits) * (1 - tf.eye(batch_size))
        log_prob = logits - tf.math.log(tf.reduce_sum(exp_logits, axis=1, keepdims=True) + 1e-12)
        mean_log_prob_pos = tf.reduce_sum(mask * log_prob, axis=1) / (tf.reduce_sum(mask, axis=1) + 1e-12)
        loss = -tf.reduce_mean(mean_log_prob_pos)
        return loss

    def _batch_sampler(self, x, y, domain_array=None, batch_size=32,
                    mode="vanilla", min_samples_per_class=2, min_samples_per_domain=1):
        batch_idx = np.random.choice(len(y), size=batch_size, replace=False)

        if mode == "vanilla":
            return x[batch_idx], y[batch_idx], domain_array[batch_idx] if domain_array is not None else None

        for cls in np.unique(y):
            cls_idx = np.where(y == cls)[0]
            n_in_batch = np.sum(y[batch_idx] == cls)
            if mode == "class_balanced" and n_in_batch < min_samples_per_class:
                n_missing = min_samples_per_class - n_in_batch
                missing_samples_pool = np.setdiff1d(cls_idx, batch_idx)
                if len(missing_samples_pool) == 0:
                    missing_samples = np.random.choice(cls_idx, n_missing, replace=True)
                else:
                    replace = len(missing_samples_pool) < n_missing
                    missing_samples = np.random.choice(missing_samples_pool, n_missing, replace=replace)
                replaceable_idx = [i for i in batch_idx if y[i] != cls]
                n_replace = min(len(replaceable_idx), len(missing_samples))
                if n_replace > 0:
                    replace_idx = np.random.choice(replaceable_idx, n_replace, replace=False)
                    batch_idx[replace_idx] = missing_samples[:n_replace]

            elif mode == "domain_aware" and domain_array is not None:
                n_missing_class = max(0, min_samples_per_class - n_in_batch)
                chosen_missing = []
                cls_domains = np.unique(domain_array[cls_idx])
                for dom in cls_domains:
                    dom_idx = cls_idx[domain_array[cls_idx] == dom]
                    n_in_dom_batch = np.sum(np.isin(batch_idx, dom_idx))
                    n_missing_dom = max(0, min_samples_per_domain - n_in_dom_batch)
                    if n_missing_dom > 0:
                        dom_pool = np.setdiff1d(dom_idx, batch_idx)
                        replace = len(dom_pool) < n_missing_dom
                        if len(dom_pool) == 0:
                            chosen_dom = np.random.choice(dom_idx, n_missing_dom, replace=True)
                        else:
                            chosen_dom = np.random.choice(dom_pool, n_missing_dom, replace=replace)
                        chosen_missing.extend(chosen_dom)
                    if len(chosen_missing) >= n_missing_class:
                        break

                if len(chosen_missing) < n_missing_class:
                    remaining = np.setdiff1d(cls_idx, np.concatenate([batch_idx, chosen_missing]))
                    n_needed = n_missing_class - len(chosen_missing)
                    if len(remaining) > 0:
                        extra = np.random.choice(remaining, n_needed, replace=True)
                        chosen_missing.extend(extra)
                    else:
                        extra = np.random.choice(chosen_missing, n_needed, replace=True)
                        chosen_missing.extend(extra)

                replaceable_idx = [i for i, idx in enumerate(batch_idx) if y[idx] != cls]
                n_replace = min(len(replaceable_idx), len(chosen_missing))
                if n_replace > 0:
                    selected_replace_idx = np.random.choice(replaceable_idx, n_replace, replace=False)
                    batch_idx[selected_replace_idx] = chosen_missing[:n_replace]

        np.random.shuffle(batch_idx)
        return x[batch_idx], y[batch_idx], domain_array[batch_idx] if domain_array is not None else None


    def train_model(self, batch_size=32, min_epochs=25, max_epochs=100, val_split=0.2, patience=10,
                domain_array=None, temperature=0.1, use_pretrained=False, pretrained_path=None,
                batch_mode="vanilla", min_samples_per_class=2, min_samples_per_domain=1):

        input_shape = self.x_train.shape[1:]
        base_network = self.create_base_network(input_shape)
        if use_pretrained and pretrained_path:
            base_network.load_weights(pretrained_path)

        x_train_main, x_val, y_train_main, y_val = train_test_split(
            self.x_train, self.y_train, test_size=val_split, stratify=self.y_train, random_state=42
        )

        optimizer = Adam(learning_rate=0.001)
        best_val_loss = np.inf
        best_weights = base_network.get_weights()
        no_improve_epochs = 0
        for epoch in range(max_epochs):
            print(f"\n=== Epoch {epoch+1}/{max_epochs} ===")
            idx = np.arange(len(x_train_main))
            np.random.shuffle(idx)
            x_train_main = x_train_main[idx]
            y_train_main = y_train_main[idx]
            domain_array_epoch = domain_array[idx] if domain_array is not None else None

            for i in range(0, len(x_train_main), batch_size):
                x_batch, y_batch, _ = self._batch_sampler(
                    x_train_main, y_train_main, domain_array=domain_array_epoch,
                    batch_size=batch_size,
                    mode=batch_mode,
                    min_samples_per_class=min_samples_per_class,
                    min_samples_per_domain=min_samples_per_domain
                )

                with tf.GradientTape() as tape:
                    features = base_network(x_batch, training=True)
                    loss = self.supcon_loss(features, y_batch, temperature=temperature)
                grads = tape.gradient(loss, base_network.trainable_variables)
                optimizer.apply_gradients(zip(grads, base_network.trainable_variables))

            val_features = base_network(x_val, training=False)
            val_loss = self.supcon_loss(val_features, y_val, temperature=temperature)
            val_loss_value = val_loss.numpy()
            print(f"Epoch {epoch+1} validation loss: {val_loss_value:.6f}")
            print(val_loss_value)
            print(best_val_loss)
            if val_loss_value < best_val_loss:
                best_val_loss = val_loss_value
                best_weights = base_network.get_weights()
                no_improve_epochs = 0
                print('no_improve_epochs', no_improve_epochs)
            else:
                no_improve_epochs += 1
                print('no_improve_epochs', no_improve_epochs)

            if epoch + 1 >= min_epochs and no_improve_epochs >= patience:
                print('epoch', epoch)
                print('min_epochs', min_epochs)
                print('no_improve_epochs', no_improve_epochs)
                print('patience', patience)
                print(f"Early stopping at epoch {epoch+1}")
                base_network.set_weights(best_weights)
                break

        train_embeddings = base_network.predict(self.x_train)
        test_embeddings = base_network.predict(self.x_test)

        return (
            base_network,
            None,
            None,
            train_embeddings,
            self.training_params,
            self.y_train,
            None,
            None
        )
