import metrics

BATCH_SIZE = 1024
IMAGE_SIZE = (224, 224)
LOGS_FOLDER = "./logs/"
CHECKPOINTS_FOLDER = "./saved_models/"
LEARNING_RATES = [1e-3, 1e-3]
EPOCHS = [10, 10]
BLOCKS_USED = 2
METRICS = [metrics.balanced_accuracy,
           metrics.accuracy,
           metrics.TP, metrics.TN,
           metrics.FP, metrics.FN,
           metrics.precision, metrics.recall]