import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from word_embedding_rl import WordEmbedding
from aggregator_predict_rl import AggPredictor
from selection_predict_rl import SelPredictor
from seq2sql_condition_predict_rl import Seq2SQLCondPredictor
# This piece of code builds the main skeleton of the model
# It calls in Lstm networks for all three (agg,sel,cond)
# Refer to the functions called here from the above imported files

class Seq2SQL(nn.Module):
    def __init__(self, word_emb, N_word, N_h=200, N_depth=2):
        super(Seq2SQL, self).__init__()

        self.N_h = N_h
        self.N_depth = N_depth

        self.max_col_num = 45 # Maximum number of column tokens
        self.max_tok_num = 200 # Maximum input length - question tokens + Sql tokens + column tokens
        self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND',
                        'EQL', 'GT', 'LT', '<BEG>']
        self.COND_OPS = ['EQL', 'GT', 'LT']

        self.embed_layer = WordEmbedding(word_emb, N_word,
                                             self.SQL_TOK, our_model=False)

        #Predict aggregator
        self.agg_pred = AggPredictor(N_word, N_h, N_depth)

        #Predict selected column
        self.sel_pred = SelPredictor(N_word, N_h, N_depth, self.max_tok_num)

        #Predict number of cond
        self.cond_pred = Seq2SQLCondPredictor(
            N_word, N_h, N_depth, self.max_col_num, self.max_tok_num)


        self.CE = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax()
        self.log_softmax = nn.LogSoftmax(dim=2)
        # self.bce_logit = nn.BCEWithLogitsLoss() #Binary_CE?
       


    def generate_gt_where_seq(self, q, col, query):
        # This generates the ground truth where sequence of tokens
        """
        :param q: question tokens = the question is tokenized as a list
                    E.g: ['what', 'station', 'aired', 'a', 'game', 'at', '9:00', 'pm', '?']
        :param col: Column header tokens = each head is a list of words it
                    contains (small letters)
                    E.g: ['Time', 'Big Ten Team'] => [['time'], ['big', 'ten', 'team']]
        :param query: the Query in tokenized form 
                    e.g ['SELECT', 'television', 'WHERE', 'time', 'EQL', '9:00', 'pm']
        
        :constant: self.SQL_TOK = ['<UNK>', '<END>', 'WHERE', 'AND', 'EQL', 'GT', 'LT', '<BEG>']

        :return ret_seq: List of indices of tokens of WHERE clauses within the token-space of the query (='all_toks') for each question
                        Start of each list is 7, end of each list is 1, any token not present in the set union(SQL_TOK, q, col) is 0
                        E.g.[[7, 2, 27, 4, 34, 1], [7, 2, 14, 15, 16, 17, 4, 38, 1]]
        """
        ret_seq = []
        for cur_q, cur_col, cur_query in zip(q, col, query):
            connect_col = [tok for col_tok in cur_col for tok in col_tok+[',']]
            all_toks = self.SQL_TOK + connect_col + [None] + cur_q + [None] #Get list of all tokens (all SQL tokens + col_names + question_tokens )
            cur_seq = [all_toks.index('<BEG>')] #initialize cur_seq
            if 'WHERE' in cur_query:
                cur_where_query = cur_query[cur_query.index('WHERE'):] #extract the condition part of the query
                cur_seq = cur_seq + [all_toks.index(tok) if tok in all_toks else 0
                                         for tok in cur_where_query] #append token indices of WHERE part of query from 'all_toks'
            cur_seq.append(all_toks.index('<END>')) #append <END> to cur_seq
            ret_seq.append(cur_seq) #
        return ret_seq


    def forward(self, q, col, col_num, pred_entry,
                gt_where = None, gt_cond=None, gt_sel=None, reinforce = False):
        B = len(q)
        pred_agg, pred_sel, pred_cond = pred_entry

        agg_score = None
        sel_score = None
        cond_score = None

        # x_emb_var = tensor containing embedded versions of all tokens in the batch;
        # x_len = number of tokens for each training example (including SQL_TOKEN, q, col). shape=(batch_size,)
        x_emb_var, x_len = self.embed_layer.gen_x_batch(q, col)

        batch = self.embed_layer.gen_col_batch(col)
        # col_len is array of number of column names for each example
        # col_name_len is array of number of tokens in each column name (across the whole batch)
        # col_inp_var is an embedding of each token in each column. shape=(col_name_len, max_tokens_in_a_column_name, embedding_vector_len)
        col_inp_var, col_name_len, col_len = batch
        # max_x_len = max(x_len) 

        if pred_agg: #forward_prop agg
            agg_score = self.agg_pred.forward(x_emb_var, x_len)

        if pred_sel: 
            sel_score = self.sel_pred.forward(x_emb_var, x_len, col_inp_var,
                                          col_name_len, col_len, col_num)

        if pred_cond:
            cond_score = self.cond_pred.forward(x_emb_var, x_len, col_inp_var,
                                            col_name_len, col_len, col_num,
                                            gt_where, gt_cond, reinforce)

        return (agg_score, sel_score, cond_score)

    def loss(self, score, truth_num, pred_entry, gt_where):
        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score
        loss = 0
        # Objective/cost function as mentioned in the paper
        # is the sum of all three losses.
        # During No Rl training, CE losses for all three are considered
        # For RL training - CE losses from the first two and rewards
        # from execution are used
        if pred_agg:
            agg_truth = [int(x[0]) for x in truth_num] #aggregator values as list
            data = torch.from_numpy(np.array(agg_truth)) #aggregator values as tensor
            agg_truth_var = Variable(data) #aggregator values as Variable

            loss += self.CE(agg_score, agg_truth_var)

        if pred_sel:
            sel_truth = [int(x[1]) for x in truth_num]
            data = torch.from_numpy(np.array(sel_truth))
            sel_truth_var = Variable(data)

            loss += self.CE(sel_score, sel_truth_var)

        if pred_cond:
            for b in range(len(gt_where)): #for each training example
                cond_truth_var = Variable(torch.from_numpy(np.array(gt_where[b][1:]))) #target value of condition after removing <BEG> token. <END> is still present as last entry.
                cond_pred_score = cond_score[b, :len(gt_where[b])-1] #extract scores for the condition. the tensor is padded with garbage values 
                loss += ( self.CE(cond_pred_score, cond_truth_var) / len(gt_where) )

        return loss

    def reinforce_backward(self, score, rewards, optimizer):
        
        scores = score[0]
        actions = torch.cat(score[1], 1)
        cur_reward = rewards[:]
        eof = self.SQL_TOK.index('<END>')
        T = len(actions[0,:])
        B = len(actions[:,0])
        actions[:,-1] = eof #Last action HAS to be one, because rewards are given at end of the episode
        eof_indices = (actions==1).max(1)[1]
        # batch_rewards = torch.zeros(64,17)
        # batch_rewards[range(B),eof_indices.data] = cur_reward
        # episode_mask = torch.ByteTensor(np.array(np.arange(T) > eof_indices.data.numpy()[:,None], dtype=np.int))
        gamma = [torch.FloatTensor([1**i for i in np.arange(t_episode)[::-1]]) for t_episode in eof_indices+1]

        discounted_rewards = torch.zeros(actions.size())
        for batch_idx, t_episode in enumerate(eof_indices.data+1):
            discounted_rewards[batch_idx, slice(t_episode)] = gamma[batch_idx] * cur_reward[batch_idx]

        neglogprobs = -self.log_softmax(scores)*Variable(discounted_rewards[:,:,None])
        entropies = (-scores * F.softmax(scores,2)).sum(-1)
        actions_neglogprobs = [neglogprobs.data[i,j,actions.data[i,j]] for i in range(B) for j in range(T)]
        entropies = entropies.sum() #summing separately to keep things clear
        
        loss = (-0.01*entropies + sum(actions_neglogprobs))/B
        # loss = sum(actions_neglogprobs)/B

        return loss
        # for t in range(T): #For each time step
        #     reward_inp = Variable(torch.FloatTensor(cur_reward))
        #     #cond_score[1][t] = cond_score[1][t].type(torch.FloatTensor)
        #     #m = (self.softmax(cond_score[1][t]))
        #     #def reinforce(var, reward):
        #         #if var.creator.reward is torch.autograd.stochastic_function._NOT_PROVIDED:
        #             #var.creator.reward = reward
        #         #else:
        #             #var.creator.reward += reward
        #     #reinforce(self, reward_inp)

        #     # creator module is changed to grad_fn in the latest
        #     #versions of torch. Therefore the module
        #     # reinforce which once could be called upon,
        #     # doesn't work anymore.
        #     actions_t = actions[t].squeeze()
        #     scores_t = scores[:, t]
            
        #     neglogprobs = -self.log_softmax(scores_t)[(range(len(actions_t)), actions_t)]
            
        #     # cond_score[1][t].grad_fn.reward = (reward_inp)
        #     #reinforce(cond_score[1][t], reward_inp)
            
        #     # for b in range(len(rewards)):
        #     #     if actions[t][b].data.numpy()[0] == eof: # "Return" is zero if reached end of episode
        #     #         cur_reward[b] = 0
        #     # advantage = reward_inp - reward_inp.mean() #Adding a baseline
        #     loss += neglogprobs.mul(reward_inp)
        #     # for b in xrange(len(actions_t)):
        #     #     optimizer.zero_grad()
        #     #     torch.autograd.backward(loss_t[b], retain_graph=True)
        #     #     # loss_[b].backward()
        #     #     optimizer.step()
            
        # return loss

    
    def check_acc(self, vis_info, pred_queries, gt_queries, pred_entry):
        # This checks the logical form accuracy of the generated query and gives
        # the total error occured. Used in calculating 
        # logical form accuracy.
        def pretty_print(vis_data):
            print('question:', vis_data[0])
            print('headers: (%s)'%(' || '.join(vis_data[1])))
            print('query:', vis_data[2])

        def gen_cond_str(conds, header):
            if len(conds) == 0:
                return 'None'
            cond_str = []
            for cond in conds:
                cond_str.append(
                    header[cond[0]] + ' ' + self.COND_OPS[cond[1]] + \
                    ' ' + str(cond[2]).lower())
            return 'WHERE ' + ' AND '.join(cond_str)

        pred_agg, pred_sel, pred_cond = pred_entry

        B = len(gt_queries)

        tot_err = agg_err = sel_err = cond_err = cond_num_err = \
                  cond_col_err = cond_op_err = cond_val_err = 0.0
        agg_ops = ['None', 'MAX', 'MIN', 'COUNT', 'SUM', 'AVG']
        for b, (pred_qry, gt_qry) in enumerate(zip(pred_queries, gt_queries)):
            good = True
            if pred_agg:
                agg_pred = pred_qry['agg']
                agg_gt = gt_qry['agg']
                if agg_pred != agg_gt:
                    agg_err += 1
                    good = False

            if pred_sel:
                sel_pred = pred_qry['sel']
                sel_gt = gt_qry['sel']
                if sel_pred != sel_gt:
                    sel_err += 1
                    good = False

            if pred_cond:
                cond_pred = pred_qry['conds']
                cond_gt = gt_qry['conds']
                flag = True
                if len(cond_pred) != len(cond_gt):
                    flag = False
                    cond_num_err += 1

                if flag and set(
                        x[0] for x in cond_pred) != set(x[0] for x in cond_gt):
                    flag = False
                    cond_col_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and cond_gt[gt_idx][1] != cond_pred[idx][1]:
                        flag = False
                        cond_op_err += 1

                for idx in range(len(cond_pred)):
                    if not flag:
                        break
                    gt_idx = tuple(x[0] for x in cond_gt).index(cond_pred[idx][0])
                    if flag and (cond_gt[gt_idx][2]) != \
                       (cond_pred[idx][2]):
                        flag = False
                        cond_val_err += 1

                if not flag:
                    cond_err += 1
                    good = False

            if not good:
                tot_err += 1

        return np.array((agg_err, sel_err, cond_err)), tot_err


    def gen_query(self, score, q, col, raw_q, raw_col, pred_entry,
                  verbose=False, reinforce = False):
    # This generates the entire query that is predicted from the 
    # model. It returns the query token list. # Which is later used
    # to calculate execution accuracy and also for Dbengine input.
        def merge_tokens(tok_list, raw_tok_str):
            tok_str = raw_tok_str.lower()
            alphabet = 'abcdefghijklmnopqrstuvwxyz0123456789$('
            special = {'-LRB-':'(', '-RRB-':')', '-LSB-':'[', '-RSB-':']',
                       '``':'"', '\'\'':'"', '--':'\u2013'}
            ret = ''
            double_quote_appear = 0
            for raw_tok in tok_list:
                if not raw_tok:
                    continue
                tok = special.get(raw_tok, raw_tok)
                if tok == '"':
                    double_quote_appear = 1 - double_quote_appear

                if len(ret) == 0:
                    pass
                elif len(ret) > 0 and ret + ' ' + tok in tok_str:
                    ret = ret + ' '
                elif len(ret) > 0 and ret + tok in tok_str:
                    pass
                elif tok == '"':
                    if double_quote_appear:
                        ret = ret + ' '
                elif tok[0] not in alphabet:
                    pass
                elif (ret[-1] not in ['(', '/', '\u2013', '#', '$', '&']) and \
                     (ret[-1] != '"' or not double_quote_appear):
                    ret = ret + ' '
                ret = ret + tok
            return ret.strip()

        pred_agg, pred_sel, pred_cond = pred_entry
        agg_score, sel_score, cond_score = score

        ret_queries = []
        if pred_agg:
            B = len(agg_score)
        elif pred_sel:
            B = len(sel_score)
        elif pred_cond:
            B = len(cond_score[0]) if reinforce else len(cond_score)
        for b in range(B):
            cur_query = {}
            if pred_agg:
                cur_query['agg'] = np.argmax(agg_score[b].data.cpu().numpy())
            if pred_sel:
                cur_query['sel'] = np.argmax(sel_score[b].data.cpu().numpy())
            if pred_cond:
                cur_query['conds'] = []
                all_toks = self.SQL_TOK + \
                           [x for toks in col[b] for x in
                            toks+[',']] + [''] + q[b] + ['']
                cond_toks = []

                if reinforce:
                    for choices in cond_score[1]:
                        if choices[b].data.cpu().numpy()[0] < len(all_toks):
                            cond_val = all_toks[choices[b].data.cpu().numpy()[0]]
                        else:
                            cond_val = '<UNK>'
                        if cond_val == '<END>':
                            break
                        cond_toks.append(cond_val)
                else:

                    for where_score in cond_score[b].data.cpu().numpy():
                        cond_tok = np.argmax(where_score)
                        cond_val = all_toks[cond_tok]
                        if cond_val == '<END>':
                            break
                        cond_toks.append(cond_val)

                if verbose:
                    print(cond_toks)
                if len(cond_toks) > 0:
                    cond_toks = cond_toks[1:]
                st = 0
                while st < len(cond_toks):
                    cur_cond = [None, None, None]
                    ed = len(cond_toks) if 'AND' not in cond_toks[st:] \
                         else cond_toks[st:].index('AND') + st
                    if 'EQL' in cond_toks[st:ed]:
                        op = cond_toks[st:ed].index('EQL') + st
                        cur_cond[1] = 0
                    elif 'GT' in cond_toks[st:ed]:
                        op = cond_toks[st:ed].index('GT') + st
                        cur_cond[1] = 1
                    elif 'LT' in cond_toks[st:ed]:
                        op = cond_toks[st:ed].index('LT') + st
                        cur_cond[1] = 2
                    else:
                        op = st
                        cur_cond[1] = 0
                    sel_col = cond_toks[st:op]
                    to_idx = [x.lower() for x in raw_col[b]]
                    pred_col = merge_tokens(sel_col, raw_q[b] + ' || ' + \
                                            ' || '.join(raw_col[b]))
                    if pred_col in to_idx:
                        cur_cond[0] = to_idx.index(pred_col)
                    else:
                        cur_cond[0] = 0
                    cur_cond[2] = merge_tokens(cond_toks[op+1:ed], raw_q[b])
                    cur_query['conds'].append(cur_cond)
                    st = ed + 1
            ret_queries.append(cur_query)

        return ret_queries
