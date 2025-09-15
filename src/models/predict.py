import numpy as np
import faiss
from src.models.inference import Infer

class Predict:
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
    
    def few_shot_predict(
        self,
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