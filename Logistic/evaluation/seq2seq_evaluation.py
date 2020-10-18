import time
import torch

# 验证过程
def evaluate(
    model,
    data_loader,
    print_every=None
    ):
    model.predict = False
    model.eval()
    if print_every == 0:
        print_every = 1

    print_loss_total = 0  # 每次打印都重置
    start = time.time()
    epoch_loss = 0
    with torch.no_grad():
        for i, batch in enumerate(data_loader):

            # shape = [seq_len, batch]
            input_batchs = batch["src"]
            target_batchs = batch["trg"]
            # list
            input_lens = batch["src_len"]
            target_lens = batch["trg_len"]

            loss = model(input_batchs, input_lens, target_batchs, target_lens, teacher_forcing_ratio=0)
            print_loss_total += loss.item()
            epoch_loss += loss.item()

            if print_every and (i+1) % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('\tCurrent Loss: %.4f' % print_loss_avg)

    return epoch_loss / len(data_loader)