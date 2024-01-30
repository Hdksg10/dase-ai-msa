import argparse
from multimodal.ALMT import ALMTConfig, ALMT
from multimodal.LMF import LMFConfig, LMF
from multimodal.MULT import MULTConfig, MULT
from multimodal.base import MultiModalConfig, MMModel
from multimodal.utils import MultiModalTrainer
from multimodal.sadatasets import TrainDataset, TestDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='lmf', help='lmf, mult, almt')
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--eval_per_epoch', action='store_true')
parser.add_argument('--modals', type=str, default='ti')

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
trainer = MultiModalTrainer(model, 
                            train_dataset, 
                            test_dataset = test_dataset, 
                            compute_metrics=accuracy_score, 
                            batch_size=args.batch_size, 
                            num_epochs=args.epochs, 
                            eval_per_epoch=args.eval_per_epoch,
                            lr=args.lr,
                            modals=args.modals)
trainer.train()
