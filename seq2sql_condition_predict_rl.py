import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Categorical
import numpy as np
from net_utils_rl import run_lstm
# This piece of code is for prediction of the WHERE conditions
class Seq2SQLCondPredictor(nn.Module):
    def __init__(self, N_word, N_h, N_depth, max_col_num, max_tok_num):
        super(Seq2SQLCondPredictor, self).__init__()
        self.N_h = N_h
        self.max_tok_num = max_tok_num # Taken 200
        self.max_col_num = max_col_num # Taken 45
        self.cond_lstm = nn.LSTM(input_size=N_word, hidden_size=N_h/2,
                num_layers=N_depth, batch_first=True,
                dropout=0.3, bidirectional=True)
        # Here we are using an encoder - decoder LSTM using pointer networks
        # Short read on Pointer Networks is available in the reference paper sections
        # of my intern folder
        # Or it is mentioned in the reference section in the Paper
        self.cond_decoder = nn.LSTM(input_size=self.max_tok_num,
                hidden_size=N_h, num_layers=N_depth,
                batch_first=True, dropout=0.3)

        self.cond_out_g = nn.Linear(N_h, N_h)
        self.cond_out_h = nn.Linear(N_h, N_h)
        self.cond_out = nn.Sequential(nn.Tanh(), nn.Linear(N_h, 1))

        self.softmax = nn.Softmax(dim=1)


    def gen_gt_batch(self, tok_seq, gen_inp=True):
        """
        :param tok_seq: (list) List of indices of tokens of WHERE clauses within the token-space of the query (='all_toks') for each question
                        Start of each list is 7, end of each list is 1, any token not present in the set union(SQL_TOK, q, col) is 0
                        E.g.[[7, 2, 27, 4, 34, 1], [7, 2, 14, 15, 16, 17, 4, 38, 1]]
        param: gen_inp: (bool)
        If gen_inp: generate the input token sequence (removing <END>)
        Otherwise: generate the output token sequence (removing <BEG>)

        :return ret_inp_var: (Variable) a one-hot-tensor with (x,y,z)=1 if example x contains token z in position y
        :return ret_len: (array) Lengths of INPUT token sequences for the decoder (without the <BEG>/<END> token)
        """
        B = len(tok_seq)
        ret_len = np.array([len(one_tok_seq)-1 for one_tok_seq in tok_seq]) 
        max_len = max(ret_len) #Longest input sequence
        ret_array = np.zeros((B, max_len, self.max_tok_num), dtype=np.float32)
        for b, one_tok_seq in enumerate(tok_seq): # for each training example
            out_one_tok_seq = one_tok_seq[:-1] if gen_inp else one_tok_seq[1:] #Don't need <END> if generating input sequence
            for t, tok_id in enumerate(out_one_tok_seq): # for each token_id in the example condition
                ret_array[b, t, tok_id] = 1 # Flag to use for decoding

        ret_inp = torch.from_numpy(ret_array)
        
        ret_inp_var = Variable(ret_inp) #[B, max_len, max_tok_num]

        return ret_inp_var, ret_len


    def forward(self, x_emb_var, x_len, col_inp_var, col_name_len, col_len,
            col_num, gt_where, gt_cond, reinforce):
        """
        :param x_emb_var: (Variable) embedded versions of ALL tokens in the batch. shape=[batch_size, max(x_len), embedding_vector_len]
        :param x_len: (array) number of tokens for each training example (including SQL_TOKEN, question_tokens, column_tokens). shape=(batch_size,)
        :param col_inp_var: (Variable) embedding of each token in each column. shape=(col_name_len, max_tokens_in_a_column_name, embedding_vector_len)
        :param col_name_len: (array) number of tokens for each column name (across the whole batch). Length = sum(col_num)
        :param col_len: (array) number of columns for each example; TODO: remove the argument, it is redundant
        :param col_num: (list) Number of columns for each example 
        :param gt_where: List of indices of tokens of WHERE clauses within the token-space of the query (='all_toks') for each question
                        Start of each list is 7, end of each list is 1, any token not present in the set union(SQL_TOK, q, col) is 0
                        E.g.[[7, 2, 27, 4, 34, 1], [7, 2, 14, 15, 16, 17, 4, 38, 1]]
        :param gt_cond: (list) the actual sets of conditions for each query [column_index, operator_index, condition]
                         e.g [[1, 0, '9:00PM'], [(some other condition if present)]]
        :param reinforce: (bool) True if RL is to be done

        :return cond_score: (Variable) shape = [batch_size, max(gt_tok_len), max_x_len]
                        Returns the score (logits) for all tokens among token classes. 
                        cond_score(x,y,z) gives score for xth example's yth token for all the z classes.
                        scores beyond y=x_len[x] (last token for that specific example) don't mean anything and are given value -100
        """
        max_x_len = max(x_len) # Max length of x in the batch
        B = len(x_len) # batch size
        mask = torch.ByteTensor(np.array(np.arange(max_x_len) >= np.array(x_len)[:,None], dtype=np.int))

        h_enc, hidden = run_lstm(self.cond_lstm, x_emb_var, x_len)
        # concat forward and backward parts of each of the states so that
        # shape of state is again (num_layers, batch_size, hidden_size)
        decoder_hidden = tuple(torch.cat((hid[:2], hid[2:]),dim=2) 
                for hid in hidden)
        
        #Teacher force training
        if gt_where is not None: #Condition contained WHERE token
            gt_tok_seq, gt_tok_len = self.gen_gt_batch(gt_where, gen_inp=True)
            g_s, _ = run_lstm(self.cond_decoder,
                    gt_tok_seq, gt_tok_len, decoder_hidden)

            h_enc_expand = h_enc.unsqueeze(1) #encoder output
            g_s_expand = g_s.unsqueeze(2) #decoder output
            cond_score = self.cond_out( self.cond_out_h(h_enc_expand) +
                    self.cond_out_g(g_s_expand) ).squeeze() #Equation 2 in paper
            
            # for each training example...
            #is training example padded on the outside?
            # if yes, set the score for padded entries to -100
            
            for idx, num in enumerate(x_len): 
                if num < max_x_len: 
                    cond_score[idx, :, num:] = -100 
        else: 
            h_enc_expand = h_enc.unsqueeze(1)
            scores = []
            choices = []
            done_set = set()
            mask = torch.ByteTensor(np.array(np.arange(max_x_len) >= np.array(x_len)[:,None], dtype=np.int))

            t = 0
            
            cur_inp = Variable(torch.zeros(B, 1, self.max_tok_num))
            cur_inp.data[:,0,7] = 1  #Set the <BEG> token
            cur_h = decoder_hidden
            while len(done_set) < B and t < 50:
                g_s, cur_h = self.cond_decoder(cur_inp, cur_h)
                g_s_expand = g_s.unsqueeze(2)

                cur_cond_score = self.cond_out(self.cond_out_h(h_enc_expand) +
                        self.cond_out_g(g_s_expand)).squeeze() #shape = (batch_size, max_x_len)
                
                cur_cond_score[mask] = -100
                scores.append(cur_cond_score)
                
                if reinforce:
                    ans_tok_var = self.softmax(cur_cond_score).multinomial() #Convert to a prob distribution for each row and then sample one value per training example
                    choices.append(ans_tok_var)
                else:
                    _, ans_tok_var = cur_cond_score.max(1).unsqueeze(1) #get highest score value and corresponding index for each row


                # Get the next sets of actions
                cur_inp = Variable(torch.zeros(B, self.max_tok_num).scatter_(1, ans_tok_var.data, 1)) # one hot
                cur_inp = cur_inp.unsqueeze(1)
                
                #Find the <END> (=1) token and add to done_set if found
                [done_set.add(a.numpy()[0]) for a in torch.nonzero(ans_tok_var.data == 1)]
                t += 1

            cond_score = torch.stack(scores, 1)
        
        if reinforce:
        	return cond_score, choices # choices are the prob action values 
            # utilized during reinforce backward - In policy Gradient method
        else:
            return cond_score