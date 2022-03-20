import json

import torch
from tqdm import tqdm
import os
import time
from tensorboardX import SummaryWriter
from copy import deepcopy
from torch.nn.init import xavier_uniform_

from utils.data_loader import prepare_data_seq
from utils.common import *
from train import *
from tensorboardX import SummaryWriter
import utils.config as config


torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)
from utils.data_reader import Lang
from baselines.transformer import Transformer
from baselines.EmoPrepend import EmoP

from Model.Empdg_G import Empdg_G
from Model.EmpDG_D import EmpDG_D

os.environ["CUDA_VISOBLE_DEVICES"] = config.device_id
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
if torch.cuda.is_available():
    torch.cuda.set_device(int(config.device_id))

def train_g(model):
    config.model = "wo_D"  # read training data for g
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size)

    for n, p in model.named_parameters():
        if p.dim() > 1 and (n != "embedding.lut.weight" and config.pretrain_emb):
            xavier_uniform_(p)

    check_iter = 2000
    try:
        if config.USE_CUDA:
            model.cuda()
        model = model.train()
        best_ppl = 1000
        patient = 0
        writer = SummaryWriter(log_dir=config.save_path)
        weights_best = deepcopy(model.state_dict())
        data_iter = make_infinite(data_loader_tra)

        for n_iter in tqdm(range(1000000)):
            loss, ppl, bce, acc = model.train_one_batch(next(data_iter), n_iter)
            writer.add_scalars('loss', {'loss_train': loss}, n_iter)
            writer.add_scalars('ppl', {'ppl_train': ppl}, n_iter)
            writer.add_scalars('bce', {'bce_train': bce}, n_iter)
            writer.add_scalars('accuracy', {'acc_train': acc}, n_iter)
            if config.noam:
                writer.add_scalars('lr', {'learning_rate': model.optimizer._rate}, n_iter)

            if (n_iter + 1) % check_iter == 0:
                model = model.eval()
                model.epoch = n_iter
                model.__id__logger = 0
                loss_val, ppl_val, bce_val, acc_val = evaluate(model, data_loader_val, ty="valid", max_dec_step=50)
                writer.add_scalars('loss', {'loss_valid': loss_val}, n_iter)
                writer.add_scalars('ppl', {'ppl_valid': ppl_val}, n_iter)
                writer.add_scalars('bce', {'bce_valid': bce_val}, n_iter)
                writer.add_scalars('accuracy', {'acc_train': acc_val}, n_iter)
                model = model.train()

                if n_iter < 13000:
                    continue
                if ppl_val <= best_ppl:
                    best_ppl = ppl_val
                    patient = 0
                    ## SAVE MODEL
                    model_save_path = os.path.join(config.save_path,
                                                   'model_{}_{:.4f}'.format(iter, best_ppl))
                    torch.save(model.state_dict(), model_save_path)
                    weights_best = deepcopy(model.state_dict())
                    print("best_ppl: {}; patient: {}".format(best_ppl, patient))
                else:
                    patient += 1
                if patient > 2: break

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    ## SAVE THE BEST
    torch.save({"models": weights_best,
                'result': [loss_val, ppl_val, bce_val, acc_val], },
               os.path.join('result/' + config.model + '_best.tar'))
    return model


def pre_train_g(model, resume=False):
    model.eval()
    try:
        if resume:
            checkpoint = torch.load('result/EmpDG_woD_best.tar', map_location=lambda storage, location: storage)
            weights_best = checkpoint['models']
            model.load_state_dict({name: weights_best[name] for name in weights_best})
        else:
            model = train_g(model)
            model.eval()
    except Exception as e:
        print('errorr...... ')

    return model


def gen_disc_data(model_g, epoch=0):
    # load data and generate predictions using model_g.
    config.model = "EmpDG"
    config.adver_train = True
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, gen_data=True)
    if config.USE_CUDA:
        model_g.cuda()
    model_g.eval()

    output_train = gen_disc_train_data(model_g, data_loader_tra, max_dec_step=30)  # obtain predicted response and its emotional words.
    print("complete training data.")
    output_dev = gen_disc_train_data(model_g, data_loader_val, max_dec_step=30)
    print("complete dev data.")
    output_test = gen_disc_train_data(model_g, data_loader_tst, max_dec_step=30)
    print("complete test data.")

    # save data
    with open("empathetic-dialogue/adver_train_data.p", "wb") as f:
        pickle.dump([output_train, output_dev, output_test], f)
        f.close()


def g_for_d(model_g, batch, adver_train=False):
    enc_batch = batch["context_batch"]
    enc_emo_batch = batch['emotion_context_batch']

    ## Semantic Understanding
    mask_semantic = enc_batch.data.eq(config.PAD_idx).unsqueeze(1)  # (bsz, src_len)->(bsz, 1, src_len)
    sem_emb_mask = model_g.embedding(batch["mask_context"])  # dialogue state  E_d
    sem_emb = model_g.embedding(enc_batch) + sem_emb_mask  # E_w+E_d
    sem_encoder_outputs = model_g.semantic_und(sem_emb, mask_semantic)  # C_u  (bsz, sem_w_len, emb_dim)

    ## Multi-resolution Emotion Perception (understanding & predicting)
    # mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
    # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
    # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
    # emo_encoder_outputs = self.emotion_pec(emo_emb, mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)
    mask_emotion = enc_emo_batch.data.eq(config.PAD_idx).unsqueeze(1)
    # emo_emb_mask = self.embedding(batch["mask_emotion_context"])
    # emo_emb = self.embedding(enc_emo_batch) + emo_emb_mask
    emo_encoder_outputs = model_g.emotion_pec(model_g.embedding(enc_emo_batch), mask_emotion)  # C_e  (bsz, emo_w_len, emb_dim)

    return sem_encoder_outputs[:, 0, :], emo_encoder_outputs[:, 0, :]


def preprocess(vocab, arr):
    """Converts words to ids."""
    sequence = [vocab.word2index[word] if word in vocab.word2index else config.UNK_idx for word in arr] + [config.EOS_idx]
    return torch.LongTensor(sequence)


def merge_two(sequence_a, sequence_b):  # len(sequences) = bsz
    lengths_a = [len(seq) for seq in sequence_a]
    lengths_b = [len(seq) for seq in sequence_b]
    max_len = max(lengths_a+lengths_b)

    # for a
    padded_seqs_a = torch.ones(len(sequence_a), max_len).long() ## padding index 1 1=True, in mask means padding.
    for i, seq in enumerate(sequence_a):
        end = lengths_a[i]
        padded_seqs_a[i, :end] = torch.LongTensor(seq[:end])

    # for b
    padded_seqs_b = torch.ones(len(sequence_b), max_len).long()  ## padding index 1 1=True, in mask means padding.
    for i, seq in enumerate(sequence_b):
        end = lengths_b[i]
        padded_seqs_b[i, :end] = torch.LongTensor(seq[:end])
    return padded_seqs_a, lengths_a, padded_seqs_b, lengths_b

def disc_batch(vocab, batch_data, pred, pred_emotion):
    trg = batch_data["target_txt"]
    trg_emotion = batch_data["target_emotion_txt"]

    # convert words into ids
    trg_list = []
    trg_emotion_list = []
    pred_list = []
    pred_emotion_list = []
    for i, p in enumerate(pred):
        pred_list.append(preprocess(vocab, p))
        pred_emotion_list.append(preprocess(vocab, pred_emotion[i]))
        trg_list.append(preprocess(vocab, trg[i]))
        trg_emotion_list.append(preprocess(vocab, trg_emotion[i]))

    # convert tensor list into tensor
    target_batch, target_lengths, pred_batch, pred_lengths = \
        merge_two(trg_list, pred_list)
    target_emotion_batch, target_emotion_lengths, pred_emotion_batch, pred_emotion_lengths = \
        merge_two(trg_emotion_list, pred_emotion_list)


    batch_data["target_batch"] = target_batch.to(config.device)
    batch_data["target_lengths"] = torch.LongTensor(target_lengths).to(config.device)
    batch_data["target_emotion_batch"] = target_emotion_batch.to(config.device)
    batch_data["target_emotion_lengths"] = torch.LongTensor(target_emotion_lengths).to(config.device)

    batch_data["pred_batch"] = pred_batch.to(config.device)
    batch_data["pred_lengths"] = torch.LongTensor(pred_lengths).to(config.device)
    batch_data["pred_emotion_batch"] = pred_emotion_batch.to(config.device)
    batch_data["pred_emotion_lengths"] = torch.LongTensor(pred_emotion_lengths).to(config.device)

    return batch_data

def pre_train_d(model_g, model_d, iters=1000, resume=True):
    config.pretrain_emb=True
    if resume:
        checkpoint = torch.load('result/D_best.tar', map_location=lambda storage, location: storage)
        model_d.load_state_dict(checkpoint)
    else:
        data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, adver_train=True)

        if config.USE_CUDA:
            model_d.cuda()
            model_g.cuda()
        model_d = model_d.train()
        model_g = model_g.eval()  # fix
        writer = SummaryWriter(log_dir="save/EmpDG_D/")

        weights_best = deepcopy(model_d.state_dict())
        data_iter = make_infinite(data_loader_tra)

        for n_iter in tqdm(range(iters)):
            # using model_g get context AND emotion context
            batch_data = next(data_iter)
            pred, pred_emotion, context, emotion_context = model_g.g_for_d(batch_data)

            # get new_batch_data
            new_batch_data = disc_batch(vocab, batch_data, pred, pred_emotion)

            # train semantic_d
            loss_d, loss_g = model_d.train_one_batch(context, emotion_context, new_batch_data, train=True)
            writer.add_scalars('loss_d', {'loss_d': loss_d}, n_iter)
            writer.add_scalars('loss_g', {'loss_g': loss_g}, n_iter)

            if n_iter % 200 == 0:
                print("Iter\tLoss_d\tLoss_g")
                print(
                    "{}\t{:.4f}\t{:.4f}".format(n_iter, loss_d, loss_g))

        model_save_path = os.path.join('result/D_best.tar')
        torch.save(model_d.state_dict(), model_save_path)

    return model_d


def adver_joint_train_gd(model_g, model_d, itr_num=5000):
    model_g.train()
    model_d.train()

    if config.USE_CUDA:
        model_g.cuda()
        model_d.cuda()
    current_step = 0
    disc_log = open("save/disc_log.txt", 'w', encoding="utf-8")
    print("==================Test performance before adver train=====================")
    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, adver_train=True, gen_data=True)
    val_loss, val_ppl, val_bce, val_acc = evaluate_disc(model_g, data_loader_tst)

    best_acc = val_acc
    patient = 0
    weights_best_g = deepcopy(model_g.state_dict())
    weights_best_d = deepcopy(model_d.state_dict())


    data_iter = make_infinite(data_loader_tra)
    try:
        for itr in range(itr_num):
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            current_step += 1
            start_time = time.time()
            for i in range(config.d_steps):  # D: 1 steps
                batch_data = next(data_iter)
                # generate data using model_g for model_d.
                pred, pred_emotion, context, emotion_context = model_g.g_for_d(batch_data)
                new_batch_data = disc_batch(vocab, batch_data, pred, pred_emotion)

                loss_d, loss_dg = model_d.train_one_batch(context, emotion_context, new_batch_data, iter_train=True)

            for i in range(config.g_steps):  # G: 5 step
                loss_g, ppl_g, bce_g, acc_g = model_g.train_one_batch(batch_data, itr, loss_from_d=loss_dg)


            if current_step % 200 == 0:
                disc_log.write("STEP {}\n".format(current_step))
                disc_log.write("Discriminator loss: {}.\n".format(loss_d))
                disc_log.write("Generator loss: {}; ppl: {}; bce: {}; acc: {}.\n".format(loss_g, ppl_g, bce_g, acc_g))

                model_g = model_g.eval()
                model_g.__id__logger = 0
                config.adver_train = True
                loss_val, ppl_val, bce_val, acc_val, d1,d2 = evaluate(model_g, data_loader_val, ty="valid", max_dec_step=50, adver_train=True)

                if acc_val > best_acc:
                    best_acc = acc_val
                    patient = 0

                    if not os.path.exists('result/adver_train/'):
                        os.makedirs('result/adver_train/')

                    ## SAVE MODEL-d
                    torch.save(model_g.state_dict(),
                               os.path.join('result/adver_train/model_g_{}_{:.4f}.tar'.format(current_step, best_acc)))
                    weights_best_g = deepcopy(model_g.state_dict())
                    ## SAVE MODEL-d
                    torch.save(model_d.state_dict(),
                               os.path.join('result/adver_train/model_d_{}_{:.4f}.tar'.format(current_step, best_acc)))
                    weights_best_d = deepcopy(model_d.state_dict())
                    print("best_acc: {}; patient: {}".format(best_acc, patient))
                    end_time = time.time()
                    print("step %d spend time: %f" % (current_step, end_time - start_time))
                else:
                    patient += 1
                if patient > 2: break
    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

    ## SAVE THE BEST
    torch.save({"models_d": weights_best_d,
                "models_g": weights_best_g,
                'result': [loss_val, ppl_val, bce_val, acc_val, d1, d2], },
               os.path.join('result/EmpDG_best.tar'))

if __name__ == '__main__':
    if not os.path.exists(config.save_path):
        os.makedirs(config.save_path)

    config.emb_dim=300
    config.label_smoothing='noam'
    config.hidden_dim=300
    config.rnn_hidden_dim=300
    config.hop=1
    config.heads=2
    config.model='EmpDG'
    config.pretrain_emb=True
    config.d_steps=1
    config.g_steps=5
    config.emb_file='vectors/glove.6B.300d.txt'



    data_loader_tra, data_loader_val, data_loader_tst, vocab, program_number = prepare_data_seq(batch_size=config.batch_size, adver_train=True)

    if config.model == "EmpDG":
        # print('=====================STEP 1: Pre-train Empathetic Generator=====================')
        model_g = Empdg_G(vocab, emotion_number=program_number)
        if config.test:

            model_g = model_g.eval()

            checkpoint = torch.load("result/EmpDG_best.tar",map_location=torch.device('cpu'))
            model_g.load_state_dict(checkpoint)
            loss_test, ppl_test, bce_test, acc_test = evaluate(model_g, data_loader_tst, ty="test", max_dec_step=50)
            print("Model: ", config.model, "End .")
        else:
            model_g = pre_train_g(model_g, resume=True)

        print('=====================STEP 2: Pre-train Discriminators==========================')
        model_d = EmpDG_D(vocab)
        model_d = pre_train_d(model_g, model_d, iters=1000, resume=config.resume_d)

        print("=====================STEP 3: Adversarial joint learning=======================") # config.resume_g is True; config.resume_d is True.
        adver_joint_train_gd(model_g, model_d, itr_num=config.adver_itr_num)
    else:
        print("end.")










