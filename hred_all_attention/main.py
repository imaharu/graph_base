''' Pytorch '''
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils.rnn import *

''' myfile'''
from model import *
from define import *
from loader import *

''' Python '''
import time
from tqdm import tqdm
from dateutil.relativedelta import relativedelta

def train(model, article_docs, summary_docs):
    loss = model(article_docs=article_docs.cuda(),
        summary_docs=summary_docs.cuda())
    loss = loss.mean()
    return loss

if __name__ == '__main__':
    print("source document length : {} ".format(len(article_data)))
    print("target document length : {} ".format(len(summary_data)))
    print("hidden_size: {} ".format(hidden_size))
    print("embed_size: {} ".format(embed_size))
    print("batch_size: {} ".format(batch_size))
    print("max_epoch: {}".format(max_epoch))
    start = time.time()
    device = "cuda:0"

    data_set = MyDataset(article_data, summary_data)
    if args.mode == "debug":
        train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater)
    else:
        train_iter = DataLoader(data_set, batch_size=batch_size, collate_fn=data_set.collater, shuffle=True)

    opts = { "bidirectional" : args.none_bid, "coverage_vector": args.coverage }
    model = Hierachical(opts)
    model.train()
    model = nn.DataParallel(model).to(device)
    print(model)

    save_dir = "{}/{}".format("trained_model", args.save_dir)
    if args.set_state:
        optimizer = torch.optim.Adagrad( model.parameters(), lr=0.15,  initial_accumulator_value=0.1)
        set_epoch = 0
    else:
        checkpoint = torch.load("trained_model/{}/{}".format(args.save_dir, args.model_path))
        max_epoch -= checkpoint['epoch']
        set_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer = torch.optim.Adagrad( model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

    for epoch in range(max_epoch):
        real_epoch = epoch + set_epoch + 1
        tqdm_desc = "[Epoch{:>3}]".format(real_epoch)
        tqdm_bar_format = "{l_bar}{bar}|{n_fmt}/{total_fmt} [{elapsed}<{remaining}]"
        tqdm_kwargs = {'desc': tqdm_desc, 'smoothing': 0.1, 'ncols': 100,
                    'bar_format': tqdm_bar_format, 'leave': False}

        for iters in tqdm(train_iter, **tqdm_kwargs):
            optimizer.zero_grad()
            loss = train(model, iters[0], iters[1])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()

        if args.mode == "train":
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
            if args.coverage:
                save_model_filename = "{}/coverage-{}.model".format(save_dir, str(real_epoch))
            else:
                save_model_filename = "{}/epoch-{}.model".format(save_dir, str(real_epoch))
            states = {
                'epoch': real_epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            torch.save(states, save_model_filename)

        elapsed_time = time.time() - start
        print("{0.days:02}日{0.hours:02}時間{0.minutes:02}分{0.seconds:02}秒".format(relativedelta(seconds=int(elapsed_time))))
