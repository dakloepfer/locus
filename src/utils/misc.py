import os
import inspect
import contextlib
import joblib
from typing import Union, List, Tuple, Dict, Any
from loguru import _Logger, logger
from itertools import chain
from argparse import ArgumentParser, _ArgumentGroup, Namespace

import torch
from yacs.config import CfgNode as CN
from pytorch_lightning.utilities import rank_zero_only
from lightning import Trainer
import cv2
import numpy as np
from einops import rearrange, repeat


def batched_2d_index_select(input_tensor, indices):
    """
    Arguments:
        input_tensor: b x c x h x w
        indices: b x n x 2 (y, x in the last dimension)
    Returns:
        output: b x n x c

    output[b,n,c] = input_tensor[b, c, indices[b,n,0], indices[b,n,1]]
    """
    b, c, h, w = input_tensor.shape
    input_tensor = rearrange(input_tensor, "b c h w -> b (h w) c")
    indices = indices[:, :, 0] * w + indices[:, :, 1]
    indices = repeat(indices, "b n -> b n c", c=c)
    output = torch.gather(input_tensor, dim=1, index=indices)
    return output


def invert_se3(T):
    """Invert an SE(3) transformation matrix."""
    assert T.shape[-2:] == (4, 4), "T must be of shape (..., 4, 4)"

    rot = T[..., :3, :3]
    trans = T[..., :3, 3]

    if type(T) == torch.Tensor:
        inv_T = torch.zeros_like(T)
        inv_rot = rot.transpose(-1, -2)
        inv_trans = torch.einsum("...ij,...j->...i", -inv_rot, trans)

    else:  # numpy
        inv_T = np.zeros_like(T)
        inv_rot = np.swapaxes(rot, -1, -2)
        inv_trans = np.einsum("...ij,...j->...i", -inv_rot, trans)

    inv_T[..., :3, :3] = inv_rot
    inv_T[..., :3, 3] = inv_trans
    inv_T[..., 3, 3] = 1.0

    return inv_T


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def apply_to_subconfig(func, cfg: CN, subcfg_name: str, **kwargs):
    subcfg_name_list = subcfg_name.strip().split(".")
    if len(subcfg_name_list) == 1:
        setattr(cfg, subcfg_name, func(getattr(cfg, subcfg_name), **kwargs))
    else:
        apply_to_subconfig(
            func,
            getattr(cfg, subcfg_name_list[0]),
            ".".join(subcfg_name_list[1:]),
            **kwargs,
        )


def make_list(x, n=-1):
    if isinstance(x, list):
        return x
    else:
        return [x for _ in range(n)]


def average_dict_list(dict_list: List[Dict[Any, Any]]) -> Dict[Any, Any]:
    """Average a list of dictionaries, recursively if needed.

    Parameters
    ----------
    dict_list (List[Dict[Any, Any]]):
        the list of dictionaries to average. All dictionaries must have the same keys.

    Returns
    -------
    Dict[Any, Any]:
        a dictionary with the same keys, but with values averaged over the list.
    """
    if len(dict_list) == 0:
        return {}

    if len(dict_list) == 1:
        return dict_list[0]

    return_dict = {}
    for k in dict_list.keys():
        if isinstance(dict_list[0][k], dict):
            return_dict[k] = average_dict_list([d[k] for d in dict_list])
        else:
            return_dict[k] = sum([d[k] for d in dict_list]) / len(dict_list)

    return return_dict


def log_on(condition, message, level):
    if condition:
        assert level in ["INFO", "DEBUG", "WARNING", "ERROR", "CRITICAL"]
        logger.log(level, message)


def get_rank_zero_only_logger(logger: _Logger):
    if rank_zero_only.rank == 0:
        return logger
    else:
        for _level in logger._core.levels.keys():
            level = _level.lower()
            setattr(logger, level, lambda x: None)
        logger._log = lambda x: None
    return logger


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument

    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))

    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def str_to_bool_or_str(val: str) -> Union[str, bool]:
    """Possibly convert a string representation of truth to bool. Returns the input otherwise. Based on the python
    implementation distutils.utils.strtobool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values are 'n', 'no', 'f', 'false', 'off', and '0'.
    """
    lower = val.lower()
    if lower in ("y", "yes", "t", "true", "on", "1"):
        return True
    if lower in ("n", "no", "f", "false", "off", "0"):
        return False
    return val


def str_to_bool(val: str) -> bool:
    """Convert a string representation of truth to bool.

    True values are 'y', 'yes', 't', 'true', 'on', and '1'; false values
    are 'n', 'no', 'f', 'false', 'off', and '0'.

    Raises:
        ValueError:
            If ``val`` isn't in one of the aforementioned true or false values.

    >>> str_to_bool('YES')
    True
    >>> str_to_bool('FALSE')
    False
    """
    val_converted = str_to_bool_or_str(val)
    if isinstance(val_converted, bool):
        return val_converted
    raise ValueError(f"invalid truth value {val_converted}")


def str_to_bool_or_int(val: str) -> Union[bool, int, str]:
    """Convert a string representation to truth of bool if possible, or otherwise try to convert it to an int.

    >>> str_to_bool_or_int("FALSE")
    False
    >>> str_to_bool_or_int("1")
    True
    >>> str_to_bool_or_int("2")
    2
    >>> str_to_bool_or_int("abc")
    'abc'
    """
    val_converted = str_to_bool_or_str(val)
    if isinstance(val_converted, bool):
        return val_converted
    try:
        return int(val_converted)
    except ValueError:
        return val_converted


def _gpus_allowed_type(x) -> Union[int, str]:
    if "," in x:
        return str(x)
    else:
        return int(x)


def _int_or_float_type(x) -> Union[int, float]:
    if "." in str(x):
        return float(x)
    else:
        return int(x)


def _get_abbrev_qualified_cls_name(cls):
    assert isinstance(cls, type), repr(cls)
    if cls.__module__.startswith("pytorch_lightning."):
        # Abbreviate.
        return f"pl.{cls.__name__}"
    else:
        # Fully qualified.
        return f"{cls.__module__}.{cls.__qualname__}"


def get_init_arguments_and_types(cls) -> List[Tuple[str, Tuple, Any]]:
    r"""Scans the class signature and returns argument names, types and default values.

    Returns:
        List with tuples of 3 values:
        (argument name, set with argument types, argument default value).

    Examples:

        >>> from pytorch_lightning import Trainer
        >>> args = get_init_arguments_and_types(Trainer)

    """
    cls_default_params = inspect.signature(cls).parameters
    name_type_default = []
    for arg in cls_default_params:
        arg_type = cls_default_params[arg].annotation
        arg_default = cls_default_params[arg].default
        try:
            arg_types = tuple(arg_type.__args__)
        except AttributeError:
            arg_types = (arg_type,)

        name_type_default.append((arg, arg_types, arg_default))

    return name_type_default


def _parse_args_from_docstring(docstring: str) -> Dict[str, str]:
    arg_block_indent = None
    current_arg = None
    parsed = {}
    for line in docstring.split("\n"):
        stripped = line.lstrip()
        if not stripped:
            continue
        line_indent = len(line) - len(stripped)
        if stripped.startswith(("Args:", "Arguments:", "Parameters:")):
            arg_block_indent = line_indent + 4
        elif arg_block_indent is None:
            continue
        elif line_indent < arg_block_indent:
            break
        elif line_indent == arg_block_indent:
            current_arg, arg_description = stripped.split(":", maxsplit=1)
            parsed[current_arg] = arg_description.lstrip()
        elif line_indent > arg_block_indent:
            parsed[current_arg] += f" {stripped}"
    return parsed


def parse_argparser(cls, arg_parser: Union[ArgumentParser, Namespace]) -> Namespace:
    """Parse CLI arguments, required for custom bool types. Copied from https://pytorch-lightning.readthedocs.io/en/1.3.8/_modules/pytorch_lightning/utilities/argparse.html#parse_argparser"""
    args = (
        arg_parser.parse_args()
        if isinstance(arg_parser, ArgumentParser)
        else arg_parser
    )

    types_default = {
        arg: (arg_types, arg_default)
        for arg, arg_types, arg_default in get_init_arguments_and_types(cls)
    }

    modified_args = {}
    for k, v in vars(args).items():
        if k in types_default and v is None:
            # We need to figure out if the None is due to using nargs="?" or if it comes from the default value
            arg_types, arg_default = types_default[k]
            if bool in arg_types and isinstance(arg_default, bool):
                # Value has been passed as a flag => It is currently None, so we need to set it to True
                # We always set to True, regardless of the default value.
                # Users must pass False directly, but when passing nothing True is assumed.
                # i.e. the only way to disable something that defaults to True is to use the long form:
                # "--a_default_true_arg False" becomes False, while "--a_default_false_arg" becomes None,
                # which then becomes True here.

                v = True

        modified_args[k] = v
    return Namespace(**modified_args)


def add_pl_argparse_args(cls, parent_parser: ArgumentParser):
    """Extends existing argparse by default attributes for a Pytorch Lightning Trainer.
    Replaces the (annoyingly deprecated) PytorchLightning function pl.Trainer.add_argparse_args().

    Code largely copied from https://pytorch-lightning.readthedocs.io/en/1.3.8/_modules/pytorch_lightning/utilities/argparse.html#add_argparse_args , with use_argument_group=True.

    Args:
        cls: a Lightning class (in practice a Trainer)
        parent_parser:
            The custom CLI arguments parser, which will be extended by the default Trainer default arguments.

    Returns:
        the parent parser with added default Trainer arguments.
    """
    parent_parser_args = [action.dest for action in parent_parser._actions]
    group_name = _get_abbrev_qualified_cls_name(cls)
    parser = parent_parser.add_argument_group(group_name)

    ignore_arg_names = ["self", "args", "kwargs"]
    if hasattr(cls, "get_deprecated_arg_names"):
        ignore_arg_names += cls.get_deprecated_arg_names()

    allowed_types = (str, int, float, bool)

    # Get symbols from cls or init function.
    for symbol in (cls, cls.__init__):
        args_and_types = get_init_arguments_and_types(symbol)
        args_and_types = [x for x in args_and_types if x[0] not in ignore_arg_names]
        if len(args_and_types) > 0:
            break

    args_help = _parse_args_from_docstring(cls.__init__.__doc__ or cls.__doc__ or "")

    for arg, arg_types, arg_default in args_and_types:
        if arg in parent_parser_args:
            continue
        arg_types = [at for at in allowed_types if at in arg_types]
        if not arg_types:
            # skip argument with not supported type
            continue
        arg_kwargs = {}
        if bool in arg_types:
            arg_kwargs.update(nargs="?", const=True)
            # if the only arg type is bool
            if len(arg_types) == 1:
                use_type = str_to_bool
            elif int in arg_types:
                use_type = str_to_bool_or_int
            elif str in arg_types:
                use_type = str_to_bool_or_str
            else:
                # filter out the bool as we need to use more general
                use_type = [at for at in arg_types if at is not bool][0]
        else:
            use_type = arg_types[0]

        if arg == "gpus" or arg == "tpu_cores":
            use_type = _gpus_allowed_type

        # hack for types in (int, float)
        if len(arg_types) == 2 and int in set(arg_types) and float in set(arg_types):
            use_type = _int_or_float_type

        # hack for track_grad_norm
        if arg == "track_grad_norm":
            use_type = float

        parser.add_argument(
            f"--{arg}",
            dest=arg,
            default=arg_default,
            type=use_type,
            help=args_help.get(arg),
            **arg_kwargs,
        )

    return parent_parser


def pl_from_argparse_args(cls, args: Union[Namespace, ArgumentParser], **kwargs):
    """Manually re-implemented function to create a Pytorch Lightning Trainer from CLI arguments, as originally used in Trainer.from_argparse_args(), which has (annoyingly) been deprecated.

    Code largely taken copied from https://pytorch-lightning.readthedocs.io/en/1.3.8/_modules/pytorch_lightning/utilities/argparse.html#from_argparse_args , with the cls argument fixed as cls=Trainer.

    Args:
        cls: a Lightning class (in practice, a Trainer)
        args: The parser or namespace to take arguments from. Only known arguments will be
            parsed and passed to the :class:`Trainer`.
        **kwargs: Additional keyword arguments that may override ones in the parser or namespace.
            These must be valid Trainer arguments.

    Returns:
        :class:`Trainer`
            Initialised Trainer object.
    """

    if isinstance(args, ArgumentParser):
        args = parse_argparser(cls, args)

    params = vars(args)

    # we only want to pass in valid Trainer args from the parser, the rest may be user specific
    valid_kwargs = inspect.signature(cls.__init__).parameters
    trainer_kwargs = dict(
        (name, params[name]) for name in valid_kwargs if name in params
    )
    trainer_kwargs.update(**kwargs)

    return cls(**trainer_kwargs)
