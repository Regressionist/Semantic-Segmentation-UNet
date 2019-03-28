import argparse
from model import *


parser = argparse.ArgumentParser(description='Trains the semantic segmentation model on cityscape dataset')
parser.add_argument('-s','--start-epoch', default=0, help='index of the first epoch')
parser.add_argument('-e','--epochs', default=10, help='number of epochs')
parser.add_argument('-a','--augment', default=0, help='ID of the data augmentation method')
parser.add_argument('-v','--version', default='v0', help='version of the model')
parser.add_argument('-w','--weights-version-load', default='v0', help='version of the model weight checkpoint to load')
parser.add_argument('-x','--weights-version-save', default='v0', help='version of the model weight checkpoint to load')
parser.add_argument('-o','--optimizer', default='adam', help="optimizer ('adam' or 'sgd')")
parser.add_argument('-l','--learning-rate', default=0.001, help='learning rate')
parser.add_argument('-d','--dropout', default=0.2, help='dropout probability')
parser.add_argument('-m','--lambda-varloss', default=100, help='multiplier for the variance loss')

args = parser.parse_args()

unet=Unet(float(args.dropout),float(args.lambda_varloss))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print (device)
unet=unet.to(device)

train_and_val(
   unet,
   start_point=int(args.start_epoch),
   end_point=int(args.epochs), 
   aug=int(args.augment), 
   version = args.version,
   weights_version_load=args.weights_version_load,
   weights_version_save=args.weights_version_save,
   optimizer=args.optimizer,
   lr=float(args.learning_rate),
   drop_p=float(args.dropout),
   lambda_varloss=float(args.lambda_varloss)
   )

print('\nversion name: ' + args.version +'\n')
