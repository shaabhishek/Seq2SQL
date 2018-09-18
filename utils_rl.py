import json
from dbengine import DBEngine
import re
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# load datasets from the data folder 
def load_data(sql_paths, table_paths):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    # max_col_num = 0
    for SQL_PATH in sql_paths:
        print("Loading data from %s"%SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                if idx >= 64:
                    break
                # sql = json.loads(line.strip(), parse_int=str, parse_float=str) #parsing everything as str so that dataloader works correctly (it can't handle multiple dtypes in the same list)
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print("Loading data from %s"%TABLE_PATH)
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab['id']] = tab

    for sql in sql_data:
        assert sql['table_id'] in table_data

    return sql_data, table_data
# Loads the sample dataset, i.e; resplit dataset
#with about 1000 in train and 100 on test
def load_dataset():
    print("Loading from re-split dataset")
    sql_data, table_data = load_data('data_resplit/train.jsonl',
                'data_resplit/tables.jsonl')
    val_sql_data, val_table_data = load_data('data_resplit/dev.jsonl',
                'data_resplit/tables.jsonl')
    test_sql_data, test_table_data = load_data('data_resplit/test.jsonl',
                'data_resplit/tables.jsonl')
    TRAIN_DB = 'data_resplit/table.db'
    DEV_DB = 'data_resplit/table.db'
    TEST_DB = 'data_resplit/table.db'

    return sql_data, table_data, val_sql_data, val_table_data,\
            test_sql_data, test_table_data, TRAIN_DB, DEV_DB, TEST_DB

def collate_fn(batch):
    # "Puts each data field into a tensor with outer dimension batch size"
    return {key: [d[key] for d in batch] for key in batch[0]}

class SQLDataset(Dataset):
    def __init__(self, sql_jsonl_file, table_jsonl_file):
        self.sql_data, self.table_data = load_data(sql_jsonl_file, table_jsonl_file)
        
        self.TRAIN_DB = 'data_resplit/table.db'
        
    def __len__(self):
        return len(self.sql_data)

    def __getitem__(self, idx):
        sql_item = self.sql_data[idx]
        sql_item_sql = sql_item['sql']

        table_id = sql_item['table_id']
        question_tokens = sql_item['question_tok']
        column_headers = self.table_data[table_id]['header_tok']
        column_num = len(self.table_data[table_id]['header'])
        sql_query = (sql_item_sql['agg'],
                     sql_item_sql['sel'],
                     len(sql_item_sql['conds']),
                     tuple(x[0] for x in sql_item_sql['conds']),
                     tuple(x[1] for x in sql_item_sql['conds']))
        query_tokens = sql_item['query_tok']
        gt_cond = sql_item_sql['conds']
        question_raw = (sql_item['question'],
                        self.table_data[table_id]['header'],
                        sql_item['query'])
        return {'question_tokens':question_tokens,
                'column_headers':column_headers,
                'column_num':column_num,
                'sql_query':sql_query,
                'query_tokens':query_tokens,
                'gt_cond':gt_cond,
                'question_raw':question_raw,
                'table_id': table_id,
                'sql_entry': sql_item_sql}


# Used to save models - can change the names as per requirements
# make sure they are properly set so your saved model
# doesn't get overwritten
def best_model_name(for_load = False):
    new_data = 'newpolicy'
    mode = 'seq2sql'
    reinforce = True
    if for_load:
        use_emb = ''
        use_rl = 'trial2'
    else:
        use_emb = ''
        use_rl = 'rl_' if reinforce else ''
    
    use_ca = ''

    agg_model_name = '%s_%s%s%s.agg_model'%(new_data,
            mode, use_emb, use_ca)
    sel_model_name = '%s_%s%s%s.sel_model'%(new_data,
            mode, use_emb, use_ca)
    cond_model_name = '%s_%s%s%s.cond_%smodel'%(new_data,
            mode, use_emb, use_ca, use_rl)
    
    return agg_model_name, sel_model_name, cond_model_name

# Used for sampling a batch of 64
def to_batch_seq(sql_data, table_data, idxes, st, ed, ret_vis_data=False):
    """
    :param idxes: (array) shuffled order of indexes
    :param st: (int) start index of idxes
    :param ed: (int) end index of idxes
    :param ret_vis_data: (bool) If True, return raw data of question, column headers and sql query for each training example

    :return q_seq: (list) question tokens = the question is tokenized as a list
    :return col_seq: (list) Column header tokens = each header is a list of words it contains (small letters)
                    E.g: ['Time', 'Big Ten Team'] => [['time'], ['big', 'ten', 'team']]

    :return col_num: (list) Number of columns for each training example

    :return ans_seq: (list) the sql queries that are generated.
                    Each example is a tuple (aggregator_idx, column_idx, num_conditions, tuple(column_idx for each condition), tuple(operator_idx for each condition))

    :return query_seq: (list) the Query in tokenized form e.g ['SELECT', 'television', 'WHERE', 'time', 'EQL', '9:00', 'pm']

    :return gt_cond_seq: (list) the actual sets of conditions for each query e.g [[1, 0, '9:00PM'], [(some other condition if present)]]

    :return vis_seq: (list) Returned if ret_vis_data is True. Each training example is returned as a
                    tuple (raw_question, list(column_headers), sql_query_generated)
    """
    q_seq = [] #question tokens = the question is tokenized as a list
    col_seq = [] #Column header tokens = each header is a list of words it contains (small letters) E.g: ['Time', 'Big Ten Team'] => [['time'], ['big', 'ten', 'team']]
    col_num = [] #Number of columns
    ans_seq = [] #the sql queries that are generated
    query_seq = [] #the Query in tokenized form e.g ['SELECT', 'television', 'WHERE', 'time', 'EQL', '9:00', 'pm']
    gt_cond_seq = [] #the actual sets of conditions for each query e.g [[1, 0, '9:00PM'], [(some other condition if present)]]
    vis_seq = [] #stores question (in natural language), relevant table_header (list of column names), and the sql query
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        q_seq.append(sql['question_tok']) #each element in sql['question_tok'] is a unicode word
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        ans_seq.append((sql['sql']['agg'],              # index of aggregator that is being used
            sql['sql']['sel'],                          # index of column that is selected
            len(sql['sql']['conds']),                   #how many conditions?
            tuple(x[0] for x in sql['sql']['conds']),   #list of column_index for each condition
            tuple(x[1] for x in sql['sql']['conds']))) #list of operator_index for each condition
        query_seq.append(sql['query_tok']) 
        gt_cond_seq.append(sql['sql']['conds'])
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query']))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq

# Samples items are taken it and the entire ground truth query is returned for
# the same
def to_batch_query(sql_data, idxes, st, ed):
    """
    Shuffle batch and return (sql query ground truth , table_id's)
    """
    query_gt = []
    table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids
# used for no Rl training - using ce losses
def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry):
    """
    :param model: model class variable
    :param optimizer: optimizer class variable
    :param sql_data: list of training entries in the dataset. each list entry is a dict
    :param table_data: dict of table data. keys are table_id's
    :param pred_entry: 3-tuple of boolean values telling which of (AGG, SEL, COND) to train

    :return cum_loss/len(sql_data): average loss
    """
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq = \
                to_batch_seq(sql_data, table_data, perm, st, ed)
        
        # gt means ground truth
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq) #token ids in where clauses
        gt_sel_seq = [x[1] for x in ans_seq] #selectors for all questions
        score = model.forward(q_seq, col_seq, col_num, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq) #returns a 3-tuple
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        
        cum_loss += loss.data.cpu().numpy()[0]*(ed - st) #cumulate loss over all training examples
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data) #average loss over training data
# checks accuracy on val/dev set
def epoch_acc(model, batch_size, sql_data, table_data, pred_entry):
    model.eval()
    perm = list(range(len(sql_data)))
    st = 0
    one_acc_num = 0.0
    tot_acc_num = 0.0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                pred_entry, gt_sel = gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data,
                pred_queries, query_gt, pred_entry)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)
# checks execution accuracy of the val/test set
def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path):
    engine = DBEngine(db_path)

    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num,
                (True, True, True), gt_sel=gt_sel_seq)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            try:
                ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            except:
                ret_gt = None
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)
        
        st = ed

    return tot_acc_num / len(sql_data)
#Used for reinforcemnt training - with two CE losses and one RL reward function
#Two loss criteria are implemented simultaneously

def epoch_reinforce_train(model, optimizer, batch_size, sql_dataloader):
    """
    :param model: (Seq2SQL class)
    :param optimizer: (optimizer object)
    :param batch_size: (int)
    :param sql_data: (list) each entry is a dict containing one training example.
                    Dict includes table_id for relevant table
    :param table_data (dict) table data dict with keys as table_id's
    :param db_path: (str) path to the table db file
    """

    # engine = DBEngine(db_path) #Init database
    engine = DBEngine(sql_dataloader.dataset.TRAIN_DB)
    model.train() #Set model in training mode
    # perm = np.random.permutation(len(sql_data))
    cum_reward = 0.0
    st = 0

    for batch_idx, sql_data in enumerate(sql_dataloader):
        gt_where_batch = model.generate_gt_where_seq(sql_data['question_tokens'], sql_data['column_headers'], sql_data['query_tokens']) #Get where clauses of examples with tokens replaced by their token_ids
        raw_q_batch = [x[0] for x in sql_data['question_raw']] # Get questions for each training example
        raw_col_batch = [x[1] for x in sql_data['question_raw']] # Get Column Headers for each training example
        gt_sel_batch = [x[1] for x in sql_data['sql_query']] # Get selector_id's for each training example
        table_ids = sql_data['table_id']
        gt_sql_entry = sql_data['sql_entry']
        score = model.forward(q=sql_data['question_tokens'],
                                col=sql_data['column_headers'],
                                col_num=sql_data['column_num'],
                                pred_entry=(True, True, True),
                                reinforce = True,
                                gt_sel=gt_sel_batch)
        loss = model.loss(score, sql_data['sql_query'], (True,True,False), gt_where_batch)
        pred_queries = model.gen_query(score, sql_data['question_tokens'], sql_data['column_headers'], raw_q_batch, raw_col_batch, (True, True, True), reinforce=True)
        
        rewards = []
        # import pdb; pdb.set_trace()
        for (sql_gt, sql_pred, tid) in zip(gt_sql_entry, pred_queries, table_ids):
            ret_gt = engine.execute(tid,
                    sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid,
                        sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None

            if ret_pred is None:
                rewards.append(-2)
            elif ret_pred != ret_gt:
                rewards.append(-1)
            else:
                rewards.append(3)

        cum_reward += (sum(rewards))
        
        lossrl = model.reinforce_backward(score[2], rewards, optimizer)
        # print("Avg RL Loss for batch: {}".format(lossrl.mean()))
        loss_batch = loss + lossrl
        
        # Optimization step batch-wise
        optimizer.zero_grad()
        loss_batch.backward(torch.FloatTensor([1]))
        optimizer.step()
        
        # Optimization step example-wise
        # for l in loss_batch:
        #     optimizer.zero_grad()
        #     l.backward(retain_graph=True)
        #     optimizer.step()

    print("Avg RL Loss for Epoch's last batch: {}. Avg CE Loss: {}".format(loss_batch.data[0], loss.data[0]))
    return cum_reward / len(sql_dataloader.dataset)

def load_word_emb(file_name):
    load_used = False
    if not load_used:
        print(('Loading word embedding from %s'%file_name))
        ret = {}
        with open(file_name) as inf:
            for idx, line in enumerate(inf):
                if (idx >= 10000):
                    break
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array([float(x) for x in info[1:]])
        return ret
    else:
        print ('Load used word embedding')
        with open('glove/word2idx.json') as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy') as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
