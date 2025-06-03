from enum import Enum
from pathlib import PosixPath
from typing import List, Optional, Tuple, TypedDict, Union, Dict, Any

import numpy as np
import torch
from PIL import Image
from torch import nn

GenericConfigDict = Dict[str, Any]


class ESServiceAction(Enum):
    DILLY_DALLY = 0
    DEC_DATA_QUALIT = 1
    INC_DATA_QUALIT = 2
    DEC_CORES = 3
    INC_CORES = 4
    DEC_PRED_QUALITY = 5
    INC_PRED_QUALITY = 6



class HabitualNetworkOutput(TypedDict):
    policy_logits: torch.Tensor
    policy_p: torch.Tensor
    policy_logp: torch.Tensor


class TransitionNetworkOutput(TypedDict):
    s: torch.Tensor
    s_dist_params: Tuple[torch.Tensor, torch.Tensor]


class WorldModelEncoding(TypedDict):
    s: Optional[torch.Tensor]
    s_dist_params: Tuple[torch.Tensor, torch.Tensor]


class MCDaciWorldModelDecOutput(TypedDict):
    o_pred: torch.Tensor


class MCDaciWorldOutput(TypedDict):
    o_pred: Optional[Any]
    s: Optional[Any]
    s_dist_params: Tuple[Any, Any]


class ExpectedFreeEnergyTerms(TypedDict):
    """
    TODO: adequate names for ig t1 and t2
    """

    pragmatic_value: torch.FloatTensor
    information_gain_t1: torch.FloatTensor
    information_gain_t2: torch.FloatTensor
    o_pred: Optional[torch.Tensor]
    s_pred_mean: Optional[torch.Tensor]
    s_pred_sampled: Optional[torch.Tensor]



class BatchForIteration(TypedDict):
    observations: torch.Tensor
    policies: torch.Tensor


class VariationalFreeEnergyT1(TypedDict):
    f_world_model: torch.FloatTensor
    log_bce_pred_o: torch.FloatTensor


class VariationalFreeEnergyT2(TypedDict):
    f_transitional: torch.FloatTensor
    analytical_kl_div: torch.FloatTensor
    analytical_kl_div_s: torch.FloatTensor


class VariationalFreeEnergyT3(TypedDict):
    f_habitual: torch.FloatTensor
    kl_div_policy_e: torch.FloatTensor
    kl_div_pi: torch.FloatTensor
    policiy_q_p: torch.FloatTensor
