NUM_WORKERS = 4                 # Number of DataLoader worker processes
BATCH_SIZE = 16                 # Number of samples per training batch
RANDOM_SEED = 42                # Seed for reproducible experiments

LEARNING_RATE = 1e-4            # Optimizer learning rate
WEIGHT_DECAY = 1e-5             # L2 regularization strength

NUM_EPOCHS = 100                # Maximum number of training epochs
EARLY_STOPPING_PATIENCE = 10    # Stop training after no validation improvement

NUM_EPOCHS_ABL = 30             # Maximum number of training epochs for ablation study
EARLY_STOPPING_ABL = 5          # Stop training after no validation improvement for ablation study

EQ_LOSS_WEIGHT = 0.7            # Weight for EQ prediction loss in AUX experiment
MSE_WEIGHT = 0.8                # Weight for EQ prediction loss
AUX_CURVE_WEIGHT = 0.2          # Weight fot AUX prediction loss in AUX experiment
SPEC_WEIGHT = 0.2               # Weight for spectral response loss
SPEC_WEIGHT_2 = 0.1             # Weight for spectral response loss in AUX experiment
GRAD_CLIP = 1.0                 # Maximum gradient norm