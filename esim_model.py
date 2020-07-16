
from torch import nn
import torch
import torch.nn.functional as F
import os
import numpy as np
from time import time
from torch.autograd import Variable
from gensim.models import Word2Vec
from sklearn.metrics import roc_auc_score
from data_process import load_esim_data_and_labels,yield_esim_data_and_labels


# 注意力机制padding部分，赋予极小的数，这样算出来的padding部分注意力分数无限接近0，减少噪音。
mask_num=-2**32+1.0
# 模型结构，标准的esim结构
class ESIM(nn.Module):
    def __init__(self,hidden_size,embeds_dim,linear_size):
        super(ESIM, self).__init__()
        self.verbose=True
        self.dropout = 0.5
        self.n_epochs=epcho
        self.learning_rate=lr
        self.optimizer_type="adam"
        self.use_cuda=True
        self.batch_size=batch_size
        self.eval_metric=roc_auc_score
        self.hidden_size = hidden_size
        self.embeds_dim = embeds_dim
        num_word = vocab_size
        self.embeds = nn.Embedding(num_word+1, self.embeds_dim,padding_idx=0)
        self.embeds.weight.data.copy_(torch.from_numpy(ce)) # 使用ce矩阵，初始化embedding矩阵。

        self.bn_embeds = nn.BatchNorm1d(self.embeds_dim)
        self.lstm1 = nn.LSTM(self.embeds_dim, self.hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(self.hidden_size*8, self.hidden_size, batch_first=True, bidirectional=True)

        self.fc = nn.Sequential(
            nn.BatchNorm1d(self.hidden_size * 8),
            nn.Linear(self.hidden_size * 8, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, linear_size),
            nn.ELU(inplace=True),
            nn.BatchNorm1d(linear_size),
            nn.Dropout(self.dropout),
            nn.Linear(linear_size, 1)
        )
    
    def soft_attention_align(self, x1, x2, mask1, mask2):
        '''
        x1: batch_size * seq_len * dim
        x2: batch_size * seq_len * dim
        '''
        # attention: batch_size * seq_len * seq_len
        attention = torch.matmul(x1, x2.transpose(1, 2))
        mask1 = mask1.float().masked_fill_(mask1, mask_num)
        mask2 = mask2.float().masked_fill_(mask2, mask_num)

        # weight: batch_size * seq_len * seq_len
        weight1 = F.softmax(attention + mask2.unsqueeze(1), dim=-1)
        x1_align = torch.matmul(weight1, x2)
        weight2 = F.softmax(attention.transpose(1, 2) + mask1.unsqueeze(1), dim=-1)
        x2_align = torch.matmul(weight2, x1)
        # x_align: batch_size * seq_len * hidden_size

        return x1_align, x2_align

    def submul(self, x1, x2):
        mul = x1 * x2
        sub = x1 - x2
        return torch.cat([sub, mul], -1)

    def apply_multiple(self, x):
        # input: batch_size * seq_len * (2 * hidden_size)
        p1 = F.avg_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        p2 = F.max_pool1d(x.transpose(1, 2), x.size(1)).squeeze(-1)
        # output: batch_size * (4 * hidden_size)
        return torch.cat([p1, p2], 1)

    def forward(self, *input):
        # batch_size * seq_len
        sent1, sent2 = input[0], input[1]
        mask1, mask2 = sent1.eq(0), sent2.eq(0)

        # embeds: batch_size * seq_len => batch_size * seq_len * dim
        x1 = self.bn_embeds(self.embeds(sent1).transpose(1, 2).contiguous()).transpose(1, 2)
        x2 = self.bn_embeds(self.embeds(sent2).transpose(1, 2).contiguous()).transpose(1, 2)

        # batch_size * seq_len * dim => batch_size * seq_len * hidden_size
        o1, _ = self.lstm1(x1)
        o2, _ = self.lstm1(x2)

        # Attention
        # batch_size * seq_len * hidden_size
        q1_align, q2_align = self.soft_attention_align(o1, o2, mask1, mask2)
        
        # Compose
        # batch_size * seq_len * (8 * hidden_size)
        q1_combined = torch.cat([o1, q1_align, self.submul(o1, q1_align)], -1)
        q2_combined = torch.cat([o2, q2_align, self.submul(o2, q2_align)], -1)

        # batch_size * seq_len * (2 * hidden_size)
        q1_compose, _ = self.lstm2(q1_combined)
        q2_compose, _ = self.lstm2(q2_combined)

        # Aggregate
        # input: batch_size * seq_len * (2 * hidden_size)
        # output: batch_size * (4 * hidden_size)
        q1_rep = self.apply_multiple(q1_compose)
        q2_rep = self.apply_multiple(q2_compose)

        # Classifier
        x = torch.cat([q1_rep, q2_rep], -1)
        similarity = self.fc(x)
        return similarity

    def fit(self,save_path = None):
        if save_path and not os.path.exists('/'.join(save_path.split('/')[0:-1])):
            print("Save path is not existed!")
            return

        if self.verbose:
            print("pre_process data ing...")
        if self.verbose:
            print("pre_process data finished")

        model = self.train()

        optimizer = torch.optim.SGD(self.parameters(), lr=self.learning_rate)
        if self.optimizer_type == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'rmsp':
            optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_type == 'adag':
            optimizer = torch.optim.Adagrad(self.parameters(), lr=self.learning_rate)

        criterion = F.binary_cross_entropy_with_logits
        max_eval=0.60
        valid_result = []
        for epoch in range(self.n_epochs):
            total_loss = 0.0
            for train_dict in yield_esim_data_and_labels("/home/kesci/shuffle_data.csv", word2id, num_to_ev*batch_size,
                                                        q_max_len=q_max_len,t_max_len=t_max_len):
                query=train_dict['q_feat_index']
                title=train_dict['t_feat_index']
                y_train=train_dict['label']
                query = np.array(query)
                y_train = np.array(y_train, dtype=np.float32)
                x_size = query.shape[0]
                batch_iter = x_size // self.batch_size
                epoch_begin_time = time()
                batch_begin_time = time()
                for i in range(batch_iter+1):
                    offset = i*self.batch_size
                    end = min(x_size, offset+self.batch_size)
                    if offset == end:
                        break
                    batch_query = Variable(torch.LongTensor(query[offset:end]))
                    batch_title = Variable(torch.LongTensor(title[offset:end]))

                    batch_y = Variable(torch.FloatTensor(y_train[offset:end]))
                    if self.use_cuda:
                        batch_query,batch_title,batch_y = batch_query.cuda(), batch_title.cuda(),batch_y.cuda()
                    optimizer.zero_grad()
                    #print(batch_query,batch_title)
                    outputs = model(batch_query,batch_title)
                    #print(outputs)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    #model = self.train()

                    total_loss += loss.item()
                    if self.verbose:
                        if i % 100 == 99:
                            eval = self.evaluate(batch_query,batch_title,batch_y)
                            print('[%d, %5d] loss: %.6f metric: %.6f time: %.1f s' %
                                  (epoch + 1, i + 1, total_loss/100.0, eval, time()-batch_begin_time))
                            total_loss = 0.0
                            batch_begin_time = time()
                            for p in model.parameters():
                                if p.grad is not None:
                                    del p.grad  # free some memory
                            torch.cuda.empty_cache()
                            model = self.train()

                query_valid = val_data["q_feat_index"]
                title_valid = val_data["t_feat_index"]
                y_valid = val_data["label"]
                query_valid = np.array(query_valid)
                title_valid = np.array(title_valid)
                y_valid = np.array(y_valid,dtype=np.float32)
                x_valid_size = query_valid.shape[0]
                valid_loss, valid_eval = self.eval_by_batch(query_valid,title_valid,y_valid, x_valid_size)
                valid_result.append(valid_eval)
                print('*' * 50)
                print('[%d] loss: %.6f val_metric: %.6f time: %.1f s' %
                      (epoch + 1, valid_loss, valid_eval,time()-epoch_begin_time))
                print('*' * 50)
                if save_path and valid_eval>max_eval:
                    max_eval=valid_eval
                    torch.save(self, save_path)
                    print("保存模型成功")
                for p in model.parameters():
                    if p.grad is not None:
                        del p.grad  # free some memory
                torch.cuda.empty_cache()
                model = self.train()
                #torch.save(self.state_dict(),save_path)

    def eval_by_batch(self,query,title,y, x_size):
        total_loss = 0.0
        y_pred = []
        batch_size = self.batch_size
        batch_iter = x_size // batch_size
        criterion = F.binary_cross_entropy_with_logits
        model = self.eval()
        for i in range(batch_iter+1):
            offset = i * batch_size
            end = min(x_size, offset + batch_size)
            if offset == end:
                break
            batch_query = Variable(torch.LongTensor(query[offset:end]))
            batch_title = Variable(torch.LongTensor(title[offset:end]))
            batch_y = Variable(torch.FloatTensor(y[offset:end]))
            if self.use_cuda:
                batch_query,batch_title,batch_y = batch_query.cuda(),batch_title.cuda(),batch_y.cuda()
            
            with torch.no_grad():
                outputs = model(batch_query,batch_title)
                pred = torch.sigmoid(outputs).cpu()
                y_pred.extend(pred.data.numpy())
                loss = criterion(outputs, batch_y)
                total_loss += loss.item()*(end-offset)
        total_metric = self.eval_metric(y,y_pred)
        return total_loss/x_size, total_metric

    def predict_proba(self, batch_q,batch_t):
        batch_q = np.array(batch_q)
        batch_t = np.array(batch_t)
        batch_q = Variable(torch.LongTensor(batch_q))
        batch_t = Variable(torch.LongTensor(batch_t))
        if self.use_cuda and torch.cuda.is_available():
            batch_q = batch_q.cuda()
            batch_t = batch_t.cuda()

        model = self.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(batch_q,batch_t)).cpu()
        return pred.data.numpy()


    def inner_predict_proba(self,batch_q,batch_t):
        model = self.eval()
        with torch.no_grad():
            pred = torch.sigmoid(model(batch_q,batch_t)).cpu()
        return pred.data.numpy()


    def evaluate(self, batch_q,batch_t,y):
        y_pred = self.inner_predict_proba(batch_q,batch_t)
        return self.eval_metric(y.cpu().data.numpy(), y_pred)
