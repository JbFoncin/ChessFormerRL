import torch as t
from torch.nn.utils.rnn import pad_sequence

from chesstools.tools import PADDING_LM_ID


def prepare_input_for_dqn_batch(inference_data_list, device='cpu', with_target=True,
                                quantile_reg=False):
    """
    generates input for training from previously generated data

    Args:
        inference_data_list (list[dict]): the list of dict to aggregate
        device (str): the device to create the tensors on
        with_target (bool): True if target is required
        quantile_reg (bool or int): False if target is a scalar. Else, target is a tensor of quantiles.
                                    Default to False

    Returns:
        dict: model inputs and targets
    """
    pieces_ids = t.cat([inf_data['pieces_ids'] for inf_data in inference_data_list], dim=0).to(device)
    colors_ids = t.cat([inf_data['colors_ids'] for inf_data in inference_data_list], dim=0).to(device)

    starting_points = [inf_data['start_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    starting_points_padded = pad_sequence(starting_points, batch_first=True, padding_value=PADDING_LM_ID).to(device)

    destinations = [inf_data['end_move_indexes'].squeeze(0) for inf_data in inference_data_list]
    destinations_padded = pad_sequence(destinations, batch_first=True, padding_value=PADDING_LM_ID).to(device)

    targets_idx = t.tensor([inf_data['target_idx'] for inf_data in inference_data_list], device=device)
    
    if quantile_reg:
        
        batch_size = len(inference_data_list)
        targets_idx = t.repeat_interleave(targets_idx, quantile_reg).view(batch_size, -1).unsqueeze(1)
        
        if with_target:
            targets = t.cat([inf_data['target'].unsqueeze(0) for inf_data in inference_data_list],
                            dim=0).squeeze(1).to(device)
        else:
            targets = None

    else:
        
        if with_target:
            targets = t.tensor([inf_data['target'] for inf_data in inference_data_list], device=device)
        else:
            targets = None

    target_mask = (starting_points_padded == PADDING_LM_ID).to(device)
    
    if quantile_reg:
        target_mask = target_mask.unsqueeze(-1).repeat(1, 1, quantile_reg)

    batch_data = {
        'model_inputs':
            {'pieces_ids': pieces_ids,
             'colors_ids': colors_ids,
             'start_move_indexes': starting_points_padded,
             'end_move_indexes': destinations_padded,
             'target_mask': target_mask},
        'targets':
            {'targets': targets,
             'targets_idx': targets_idx}
    }

    return batch_data