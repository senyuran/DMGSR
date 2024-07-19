from option import args
import torch
import utility
import os
import data
import model
import loss
# from trainer import Trainer
# from trainer_noHR import Trainer
from trainer_NOHR_DMGSR import Trainer


if __name__ == '__main__':
    torch.manual_seed(args.seed)
    checkpoint = utility.checkpoint(args)
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["CUDA_VISIBLE_DEVICES"] = "7,6,1,0,3,2,5,4"
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)
        while not t.terminate():
            t.test()

        checkpoint.done()
