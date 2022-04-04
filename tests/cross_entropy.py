import torch
from torch.nn import CrossEntropyLoss

BATCH_SIZE = 2
MAX_SEQ_LENGTH = 3
EMBEDDING_SIZE = VOCAB_SIZE = 5

input_ids = torch.randn((BATCH_SIZE, EMBEDDING_SIZE, MAX_SEQ_LENGTH))

labels = torch.randint(low=0, high=4, size=(BATCH_SIZE, MAX_SEQ_LENGTH))
loss_fct = CrossEntropyLoss(reduction='none')

print(loss_fct(input=input_ids, target=labels).shape)  # BATCH_SIZE, MAX_SEQ_LENGTH
