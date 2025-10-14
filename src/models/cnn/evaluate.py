from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from src.utils.eval import save_confusion_matrix, save_classification_report

import logging
log = logging.getLogger(__name__)

def eval_cnn(y_pred, y_test, results_dir, label_encoder, identifier):
    
    label_mapping = dict(enumerate(label_encoder.classes_))

    acc = accuracy_score(y_test, y_pred)
    
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    log.info(f"CNN Accuracy: {100 * acc:.2f}%")
    log.info(f"CNN Classification Report:\n{report}")
    log.info(f"CNN Confusion Matrix:\n{cm}")
    save_confusion_matrix(results_dir, identifier, cm, label_mapping)
    save_classification_report(results_dir, identifier, y_test, y_pred, label_mapping)
    
    return acc, report, cm