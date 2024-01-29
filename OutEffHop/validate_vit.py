#!/usr/bin/env python3
""" ImageNet Validation Script

This is intended to be a lean and easily modifiable ImageNet validation script for evaluating pretrained
models or training checkpoints against ImageNet or similarly organized image datasets. It prioritizes
canonical PyTorch, standard Python style, and good performance. Repurpose as you see fit.

Hacked together by Ross Wightman (https://github.com/rwightman)
"""
import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model
    
from transformers_language.args import parse_args
from transformers_language.dataset_setups import DatasetSetups
from transformers_language.models.vit_attention import (
    AttentionGateType,
    ViTSelfAttentionWithExtras,
)
from transformers_language.models.softmax import SOFTMAX_MAPPING

from transformers_language.utils import (
    count_params,
    kurtosis,
    pass_data_for_range_estimation,
    val_qparams,
)

EXTRA_METRICS = True

try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')


logger = logging.getLogger("validate_vit")
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)


scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')
parser.add_argument('--report_to', default='wandb', type=str,
                   help='reporting destination (default: "wandb"')
parser.add_argument('--with_tracking', action='store_true', default=False,
                   help='enable wandb tracking')
parser.add_argument('--attn_softmax', default='vanilla', type=str)
parser.add_argument('--run_name', default='', type=str)

parser.add_argument("--fine_tuning", action="store_true")
parser.add_argument(
    "--skip_attn",
    action="store_true",
    help="Skip attention (don't update the residual).",
)
parser.add_argument('--output_dir', default='', type=str)
parser.add_argument(
    "--max_seq_length",
    type=int,
    default=None,
    help=(
        "The maximum total input sequence length after tokenization. Sequences longer than "
        "this will be truncated."
    ),
)
parser.add_argument(
        "--attn_gate_type",
        type=str,
        default=AttentionGateType.none.name,
        help="The type of gating to use for the self-attention.",
        choices=AttentionGateType.list_names(),
    )
parser.add_argument(
        "--attn_gate_init",
        type=float,
        default=0.5,
        help="init bias s.t. the gate prob is approx this value",
    )
parser.add_argument(
        "--attn_gate_mlp",
        action="store_true",
        help="Use MLP instead of single linear layer to predict the gate.",
    )
parser.add_argument(
        "--attn_gate_mlp2",
        action="store_true",
        help="Use bigger MLP instead of single linear layer to predict the gate.",
    )
parser.add_argument(
        "--attn_gate_linear_all_features",
        action="store_true",
        help="Use Linear (d_model -> n_heads) instead of n_heads Linear's (d_head -> 1).",
    )
parser.add_argument(
        "--gamma",
        type=float,
        default=None,
        help="If specified, use clipped softmax gamma",
    )
parser.add_argument(
    '--tb_scalar_log_interval',
    type=int,
    default=1000,
)
parser.add_argument(
    '--seed',
    type=int,
    default=1000,
)
# if args.seed is not None:
#         set_seed(args.seed)

def attach_act_hooks(model):
    act_dict = OrderedDict()

    def _make_hook(name):
        def _hook(mod, inp, out):
            if isinstance(inp, tuple) and len(inp) > 0:
                inp = inp[0]
            act_dict[name] = (inp, out)

        return _hook

    for name, module in model.named_modules():
        module.register_forward_hook(_make_hook(name))
    return act_dict

def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    # resolve AMP arguments based on PyTorch / Apex availability
    use_amp = None
    amp_autocast = suppress
    if args.amp:
        if args.amp_impl == 'apex':
            assert has_apex, 'AMP impl specified as APEX but APEX is not installed.'
            assert args.amp_dtype == 'float16'
            use_amp = 'apex'
            logger.info('Validating in mixed precision with NVIDIA APEX AMP.')
        else:
            assert has_native_amp, 'Please update PyTorch to a version with native AMP (or use APEX).'
            assert args.amp_dtype in ('float16', 'bfloat16')
            use_amp = 'native'
            amp_dtype = torch.bfloat16 if args.amp_dtype == 'bfloat16' else torch.float16
            amp_autocast = partial(torch.autocast, device_type=device.type, dtype=amp_dtype)
            logger.info('Validating in mixed precision with native PyTorch AMP.')
    else:
        logger.info('Validating in float32. AMP not enabled.')

    if args.fuser:
        set_jit_fuser(args.fuser)

    if args.fast_norm:
        set_fast_norm()

    # create model
    in_chans = 3
    if args.in_chans is not None:
        in_chans = args.in_chans
    elif args.input_size is not None:
        in_chans = args.input_size[0]

    model = create_model(
        args.model,
        pretrained=args.pretrained,
        num_classes=args.num_classes,
        in_chans=in_chans,
        global_pool=args.gp,
        scriptable=args.torchscript,
        **args.model_kwargs,
    )
    if args.num_classes is None:
        assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
        args.num_classes = model.num_classes

    if args.checkpoint:
        load_checkpoint(model, args.checkpoint, args.use_ema)

    if args.reparam:
        model = reparameterize_model(model)

    # Mimic validate_mlm.py to replace Self-attention module with ours
    for layer_idx in range(len(model.blocks)):
        old_self = model.blocks[layer_idx].attn
        new_self = ViTSelfAttentionWithExtras(
            # inherit from old self-attention
            num_heads=old_self.num_heads,
            dim=old_self.num_heads * old_self.head_dim,
            qk_norm = False,
            attn_drop=0.,
            proj_drop=0.,  
            softmax_fn=SOFTMAX_MAPPING[args.attn_softmax],
            gamma=args.gamma,
            skip_attn=args.skip_attn,
            attn_gate_type=AttentionGateType[args.attn_gate_type],
            attn_gate_init=args.attn_gate_init,
            attn_gate_mlp=args.attn_gate_mlp,
            attn_gate_mlp2=args.attn_gate_mlp2,
            attn_gate_linear_all_features=args.attn_gate_linear_all_features,
            fine_tuning=args.fine_tuning,
            max_seq_length=args.max_seq_length,
        )

        # copy loaded weights
        new_self.load_state_dict(old_self.state_dict(), strict=False)
        model.blocks[layer_idx].attn = new_self

    # Gating -> load the model again to load missing alpha
    if args.attn_gate_type != "none":
        path = args.checkpoint + "last.pth.tar"
        state_dict = torch.load(path)
        new_state_dict = {}
        for name, val in state_dict.items():
            if "alpha" in name:
                new_state_dict[name] = val
        model.load_state_dict(new_state_dict, strict=False)



    n_embeddings = count_params(model.pos_embed)
    n_encoder = count_params(model.blocks)
    n_head = count_params(model.head)
    logger.info(
        f"\nNumber of parameters:\n"
        f"\t* Embeddings:\t{n_embeddings}\n"
        f"\t* Encoder:\t{n_encoder}\n"
        f"\t* Head:\t{n_head}\n"
        f"\t= Total (pre-training):\t{n_embeddings + n_encoder + n_head}\n"
        f"\t= Total (encoder):\t{n_embeddings + n_encoder}\n"
    )

    param_count = sum([m.numel() for m in model.parameters()])
    logger.info('Model %s created, param count: %d' % (args.model, param_count))

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )
    test_time_pool = False
    if args.test_pool:
        model, test_time_pool = apply_test_time_pool(model, data_config)

    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    if args.torchscript:
        assert not use_amp == 'apex', 'Cannot use APEX AMP with torchscripted model'
        model = torch.jit.script(model)
    elif args.torchcompile:
        assert has_compile, 'A version of torch w/ torch.compile() is required for --compile, possibly a nightly.'
        torch._dynamo.reset()
        model = torch.compile(model, backend=args.torchcompile)
    elif args.aot_autograd:
        assert has_functorch, "functorch is needed for --aot-autograd"
        model = memory_efficient_fusion(model)

    if use_amp == 'apex':
        model = amp.initialize(model, opt_level='O1')

    if args.num_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(args.num_gpu)))

    criterion = nn.CrossEntropyLoss().to(device)

    root_dir = args.data or args.data_dir
    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
    )

    if args.valid_labels:
        with open(args.valid_labels, 'r') as f:
            valid_labels = [int(line.rstrip()) for line in f]
    else:
        valid_labels = None

    if args.real_labels:
        real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    else:
        real_labels = None

    crop_pct = 1.0 if test_time_pool else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config['crop_mode'],
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    
    
    # if args.quantize:
    #     click_config = get_quant_config()

    #     # override number of batches
    #     click_config.act_quant.num_batches = args.est_num_batches
    #     click_config.quant.n_bits = args.n_bits
    #     click_config.quant.n_bits_act = args.n_bits_act
    #     if args.no_weight_quant:
    #         click_config.quant.weight_quant = False
    #     if args.no_act_quant:
    #         click_config.quant.act_quant = False

    #     # Weight Ranges
    #     if args.ranges_weights == "minmax":
    #         pass
    #     elif args.ranges_weights in ("mse", "MSE"):
    #         click_config.quant.weight_quant_method = RangeEstimators.MSE
    #         click_config.quant.weight_opt_method = OptMethod.grid
    #     else:
    #         raise ValueError(f"Unknown weight range estimation: {args.ranges_weights}")

    #     # Acts ranges
    #     if args.percentile is not None:
    #         click_config.act_quant.options["percentile"] = args.percentile

    #     if args.ranges_acts == "running_minmax":
    #         click_config.act_quant.quant_method = RangeEstimators.running_minmax

    #     elif args.ranges_acts == "MSE":
    #         click_config.act_quant.quant_method = RangeEstimators.MSE
    #         if args.qmethod_acts == "symmetric_uniform":
    #             click_config.act_quant.options = dict(opt_method=OptMethod.grid)
    #         elif args.qmethod_acts == "asymmetric_uniform":
    #             click_config.act_quant.options = dict(opt_method=OptMethod.golden_section)

    #     elif args.ranges_acts.startswith("L"):
    #         click_config.act_quant.quant_method = RangeEstimators.Lp
    #         p_norm = float(args.ranges_acts.replace("L", ""))
    #         options = dict(p_norm=p_norm)
    #         if args.qmethod_acts == "symmetric_uniform":
    #             options["opt_method"] = OptMethod.grid
    #         elif args.qmethod_acts == "asymmetric_uniform":
    #             options["opt_method"] = OptMethod.golden_section
    #         click_config.act_quant.options = options

    #     else:
    #         raise NotImplementedError(f"Unknown act range estimation setting, '{args.ranges_acts}'")

    #     qparams = val_qparams(click_config)
    #     qparams["quant_dict"] = {}

    #     model = QuantizedBertForMaskedLM(model, **qparams)
    #     model.set_quant_state(
    #         weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
    #     )

    #     logger.info("Quantized model:")
    #     logger.info(model)

    #     # Range estimation
    #     logger.info("** Estimate quantization ranges on training data **")
    #     pass_data_for_range_estimation(
    #         loader=eval_dataloader,
    #         model=model,
    #         act_quant=click_config.quant.act_quant,
    #         max_num_batches=click_config.act_quant.num_batches,
    #     )
    #     model.fix_ranges()
    #     model.set_quant_state(
    #         weight_quant=click_config.quant.weight_quant, act_quant=click_config.quant.act_quant
    #     )
    
    act_dict = {}
    if EXTRA_METRICS:
        act_dict = attach_act_hooks(model)
        
    num_layers = model.depth
    act_inf_norms = OrderedDict()
    act_kurtoses = OrderedDict()
    act_kurtoses_ffn = OrderedDict()
    model.eval()
    
    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5
                    )
                )
            if EXTRA_METRICS:
                for j in range(num_layers):
                    for name, module in model.module.named_modules():
                        x_inp, x_out = act_dict[name]

                        x = x_out

                        # inf-norm
                        x = x.view(x.size(0), -1)
                        inf_norms = x.norm(dim=1, p=np.inf)
                        if not name in act_inf_norms:
                            act_inf_norms[name] = AverageMeter()
                        for v in inf_norms:
                            act_inf_norms[name].update(v.item())

                        # kurtosis
                        if batch_idx <= 256:
                            kurt = kurtosis(x)
                            if not name in act_kurtoses:
                                act_kurtoses[name] = AverageMeter()
                            for v in kurt:
                                act_kurtoses[name].update(v.item())
                                
                            for name in (f"blocks.{j}.mlp.fc2", f"blocks.{j}"):
                                x_inp, x_out = act_dict[name]
                                x = x_out
                                kurt = kurtosis(x)
                                if not name in act_kurtoses:
                                    act_kurtoses_ffn[name] = AverageMeter()
                                for v in kurt:
                                    act_kurtoses_ffn[name].update(v.item())
                                
                                
                            
                                
                        # # compute inf norm also for input
                        # if "norm1" in name or "norm2" in name:
                        #     x = x_inp
                        #     x = x.view(x.size(0), -1)
                        #     inf_norms = x.norm(dim=1, p=np.inf)
                        #     name += ".input"
                        #     if not name in act_inf_norms:
                        #         act_inf_norms[name] = AverageMeter()
                        #     for v in inf_norms:
                        #         act_inf_norms[name].update(v.item())


    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'],
    )

    logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    if EXTRA_METRICS:
        for name, v in act_inf_norms.items():
            results[name] = v.avg
            
        # max_inf_norm = max(v.avg for v in act_inf_norms.values())
        max_ffn_out_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "mlp" in k)
        max_layer_inf_norm = max(
            act_inf_norms[f"blocks.{j}"].avg for j in range(num_layers)
        )
        #max_LN_inp_inf_norm = max(v.avg for k, v in act_inf_norms.items() if "input" in k)
        # for name in (
        #     f"blocks.{j}.mlp.fc2",  # FFN output 
        #     ):
        #     act_kurtoses_ffn = act_kurtoses[name]
        #     avg_kurtosis = sum(v.avg for v.value in 
            
        avg_kurtosis = sum(v.avg for v in act_kurtoses.values()) / len(act_kurtoses.values())
        avg_kurtosis_ffn = sum(v.avg for v in act_kurtoses_ffn.values()) / len(act_kurtoses_ffn.values())
        max_kurtosis = max(v.avg for v in act_kurtoses.values())

        # results["max_inf_norm"] = max_inf_norm 
        results["max_ffn_out_inf_norm"] = max_ffn_out_inf_norm
        results["max_layer_inf_norm"] = max_layer_inf_norm
        #results["max_LN_inp_inf_norm"] = max_LN_inp_inf_norm
        results["avg_kurtosis"] = avg_kurtosis
        results["avg_kurtosis_ffn"] = avg_kurtosis_ffn
        results["max_kurtosis"] = max_kurtosis

        # logger.info(f"max inf norm: {max_inf_norm:.1f}")
        logger.info(f"max FFN output inf norm: {max_ffn_out_inf_norm:.1f}")
        logger.info(f"max layer inf norm: {max_layer_inf_norm:.1f}")
        # logger.info(f"max LN(FFN i + o) inf norm: {max_LN_out_inf_norm:.1f}")
        logger.info(f"Avg Kurtosis: {avg_kurtosis:.2f}")
        logger.info(f"Avg Kurtosis for ffn and final output: {avg_kurtosis_ffn:.2f}")
        logger.info(f"Max Kurtosis: {max_kurtosis:.1f}")

    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "all_results.json"), "w") as f:
            json.dump(results, f)
    
    print(results)
    return results


def _try_run(args, initial_batch_size):
    batch_size = initial_batch_size
    results = OrderedDict()
    error_str = 'Unknown'
    while batch_size:
        args.batch_size = batch_size * args.num_gpu  # multiply by num-gpu for DataParallel case
        try:
            if torch.cuda.is_available() and 'cuda' in args.device:
                torch.cuda.empty_cache()
            results = validate(args)
            return results
        except RuntimeError as e:
            error_str = str(e)
            logger.error(f'"{error_str}" while running validation.')
            if not check_batch_size_retry(error_str):
                break
        batch_size = decay_batch_step(batch_size)
        logger.warning(f'Reducing batch size to {batch_size} for retry.')
    results['error'] = error_str
    logger.error(f'{args.model} failed to validate ({error_str}).')
    return results


_NON_IN1K_FILTERS = ['*_in21k', '*_in22k', '*in12k', '*_dino', '*fcmae', '*seer']


def main():
    setup_default_logging()
    args = parser.parse_args()
    model_cfgs = []
    model_names = []
    if os.path.isdir(args.checkpoint):
        # validate all checkpoints in a path with same model
        checkpoints = glob.glob(args.checkpoint + '/*.pth.tar')
        checkpoints += glob.glob(args.checkpoint + '/*.pth')
        model_names = list_models(args.model)
        model_cfgs = [(args.model, c) for c in sorted(checkpoints, key=natural_key)]
    else:
        if args.model == 'all':
            # validate all models in a list of names with pretrained checkpoints
            args.pretrained = True
            model_names = list_models(
                pretrained=True,
                exclude_filters=_NON_IN1K_FILTERS,
            )
            model_cfgs = [(n, '') for n in model_names]
        elif not is_model(args.model):
            # model name doesn't exist, try as wildcard filter
            model_names = list_models(
                args.model,
                pretrained=True,
            )
            model_cfgs = [(n, '') for n in model_names]

        if not model_cfgs and os.path.isfile(args.model):
            with open(args.model) as f:
                model_names = [line.rstrip() for line in f]
            model_cfgs = [(n, None) for n in model_names if n]

    if len(model_cfgs):
        logger.info('Running bulk validation on these pretrained models: {}'.format(', '.join(model_names)))
        results = []
        try:
            initial_batch_size = args.batch_size
            for m, c in model_cfgs:
                args.model = m
                args.checkpoint = c
                r = _try_run(args, initial_batch_size)
                if 'error' in r:
                    continue
                if args.checkpoint:
                    r['checkpoint'] = args.checkpoint
                results.append(r)
        except KeyboardInterrupt as e:
            pass
        results = sorted(results, key=lambda x: x['top1'], reverse=True)
    else:
        if args.retry:
            results = _try_run(args, args.batch_size)
        else:
            results = validate(args)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()



if __name__ == '__main__':
    main()