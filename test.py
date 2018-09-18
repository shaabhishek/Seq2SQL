from utils_rl import SQLDataset
from torch.utils.data import DataLoader


    # elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
    #         and elem_type.__name__ != 'string_':
    #     elem = batch[0]
    #     if elem_type.__name__ == 'ndarray':
    #         # array of string classes and object
    #         if re.search('[SaUO]', elem.dtype.str) is not None:
    #             raise TypeError(error_msg.format(elem.dtype))

    #         return torch.stack([torch.from_numpy(b) for b in batch], 0)
    #     if elem.shape == ():  # scalars
    #         py_type = float if elem.dtype.name.startswith('float') else int
    #         return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    # elif isinstance(batch[0], int):
    #     return torch.LongTensor(batch)
    # elif isinstance(batch[0], float):
    #     return torch.DoubleTensor(batch)
    # elif isinstance(batch[0], string_classes):
    #     return batch
    # elif isinstance(batch[0], collections.Mapping):
    #     return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    # elif isinstance(batch[0], collections.Sequence):
    #     transposed = zip(*batch)
    #     return [default_collate(samples) for samples in transposed]

    # raise TypeError((error_msg.format(type(batch[0])))))

sq = SQLDataset('data_resplit/train.jsonl', 'data_resplit/tables.jsonl')
sql_dataloader = DataLoader(sq, batch_size=16, num_workers=3, collate_fn=collate_fn)
import pdb; pdb.set_trace()
# print(sq[0])
for idx, dat in enumerate(sql_dataloader):
    print(dat)