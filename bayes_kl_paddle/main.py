import yaml
import argparse
from data import prep_dataset,prep_loader
from utils import same_seeds
from train import train_model
from eval import evaluate_model
from predict import predict
from model import build_model
from utils import logger,metrics

def main(args):
    conf = yaml.load(open(args.config, 'r', encoding='utf-8'), Loader=yaml.FullLoader)
    same_seeds(seed=conf['hparas']['seeds'])
    logger.info('Prep | Loading datasets...')
    train_set, test_set, in_channels, num_classes = prep_dataset(conf)
    train_loader, valid_loader, test_loader = prep_loader(conf,train_set, test_set)
    logger.info(f'Prep | Train num:{len(train_loader.dataset)} | Val num:{len(valid_loader.dataset)} | Test num:{len(test_set)}')
    logger.info('Prep | Loading model...')
    model=build_model(conf)
    if args.mode=='train':
        logger.info('Train | Training...')
        train_model(conf,model,train_loader,valid_loader)
    elif args.mode=='eval':
        logger.info('Eval | Evaluating...')
        criterion = metrics.ELBO(len(train_loader.dataset))
        val_loss,val_acc=evaluate_model(conf,model, criterion, test_loader)
        logger.info(f'Eval | Test loss: {float(val_loss)} | Test acc:{float(val_acc)} | Test error:{100.-float(val_acc)*100.}')
    elif args.mode=='pred':
        res=predict(conf,model,test_loader)
        logger.info(f'result is: \n {res}')
    else:
        logger.info('Mode error!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Paddle Bayesian Model Training', add_help=True)
    parser.add_argument('-c', '--config', default='config/base.yaml', type=str, metavar='FILE', help='yaml file path')
    parser.add_argument('-m', '--mode', default='train', type=str, choices=['train', 'eval', 'pred'])
    args = parser.parse_args()
    main(args)

