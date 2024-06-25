import torch

def concordance_index_torch(score, time_value, event):
    '''
    Args
        score: 		predicted score, tensor of shape (None, )
        time_value:		true survival time_value, tensor of shape (None, )
        event:		event, tensor of shape (None, )
    '''

    ## find index pairs (i,j) satisfying time_value[i]<time_value[j] and event[i]==1
    ix1,ix2 = torch.where(
        torch.logical_and(time_value.unsqueeze(-1)<time_value,(event>0).unsqueeze(-1)))

    ## count how many score[i]<score[j]
    s1 = torch.gather(score,0, ix1)
    s2 = torch.gather(score,0, ix2)
    ci = torch.mean((s1 < s2).to(torch.float32))

    return ci