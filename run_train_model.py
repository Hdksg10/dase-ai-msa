import argparse
from multimodal.ALMT import ALMTConfig, ALMT
from multimodal.LMF import LMFConfig, LMF
from multimodal.MULT import MULTConfig, MULT
from multimodal.base import MultiModalConfig, MMModel
from multimodal.utils import MultiModalTrainer
from multimodal.sadatasets import TrainDataset, TestDataset
from sklearn.metrics import accuracy_score, mean_absolute_error, f1_score
import pandas as pd
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='almt', help='model to run, default almt, almt for ALMT, lmf for LMF, mult for MULT, base for base model(naive concat)')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train, default 10')
parser.add_argument('--batch_size', type=int, default=32, help='batch size for training, default 32')
parser.add_argument('--lr', type=float, default=1e-4, help='learning rate, default 1e-4')
parser.add_argument('--eval_per_epoch', action='store_true', help='whether to evaluate on validation dataset per epoch')
parser.add_argument('--modals', type=str, default='ti', help='modals to use, default ti, ti for text and image, i for image only, t for text only')
parser.add_argument('--predict', action='store_true', help='whether to predict on test dataset')
parser.add_argument('--predict_path', type=str, default='predict.csv', help='path to save predict result')

args = parser.parse_args()

train_dataset = TrainDataset('./datasets')
test_dataset = TestDataset('./datasets')

if args.model == 'lmf':
    config = LMFConfig(3)
    model = LMF(config)
elif args.model == 'mult':
    config = MULTConfig(3)
    model = MULT(config)
elif args.model == 'almt':
    config = ALMTConfig(3)
    model = ALMT(config)
elif args.model == 'base':
    config = MultiModalConfig(3)
    model = MMModel(config)
# print(args)
compute_metrics = [accuracy_score, mean_absolute_error, f1_score]
trainer = MultiModalTrainer(model, 
                            train_dataset, 
                            test_dataset = test_dataset, 
                            compute_metrics=compute_metrics, 
                            batch_size=args.batch_size, 
                            num_epochs=args.epochs, 
                            eval_per_epoch=args.eval_per_epoch,
                            lr=args.lr,
                            modals=args.modals,)
trainer.train()
if args.predict:
    predicts = trainer.predict()
    data = pd.DataFrame(predicts)
    print(data)
    data.to_csv(args.predict_path, index=False)