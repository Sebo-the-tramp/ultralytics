# Imports from 
# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

from ultralytics.nn.modules import (AIFI, C1, C2, C3, C3TR, SPP, SPPF, Bottleneck, BottleneckCSP, C2f, C3Ghost, C3x,
                                    Classify, Concat, Conv, Conv2, ConvTranspose, Detect, DWConv, DWConvTranspose2d,
                                    Focus, GhostBottleneck, GhostConv, HGBlock, HGStem, Pose, RepC3, RepConv,
                                    RTDETRDecoder, Segment)
from ultralytics.yolo.utils import DEFAULT_CFG_DICT, DEFAULT_CFG_KEYS, LOGGER, colorstr, emojis, yaml_load
from ultralytics.yolo.utils.checks import check_requirements, check_suffix, check_yaml
from ultralytics.yolo.utils.torch_utils import (fuse_conv_and_bn, fuse_deconv_and_bn, initialize_weights,
                                                intersect_dicts, make_divisible, model_info, scale_img, time_sync)


from micromind import PhiNet

try:
    import thop
except ImportError:
    thop = None

def get_output_dim(data_config, model):     
    """Returns intermediate representations shapes."""     
    x = torch.randn(*[1] + list(data_config["input_size"]))
    out_dim = [model._layers[0](x)]     
    names = [model._layers[0].__class__]     
    for layer in model._layers[1:]:         
        out_dim.append(layer(out_dim[-1]))
        names.append(layer.__class__)
    return [list(o.shape)[1:] for o in out_dim], names

def parse_model(d, ch,verbose=True):
    # Parse a YOLO model.yaml dictionary into a PyTorch model
    import ast

    # Args
    max_channels = float('inf')
    nc, act, scales = (d.get(x) for x in ('nc', 'act', 'scales'))
    depth, width, kpt_shape = (d.get(x, 1.0) for x in ('depth_multiple', 'width_multiple', 'kpt_shape'))
    if scales:
        scale = d.get('scale')
        if not scale:
            scale = tuple(scales.keys())[0]
            LOGGER.warning(f"WARNING ⚠️ no model scale passed. Assuming scale='{scale}'.")
        depth, width, max_channels = scales[scale]

    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {act}")  # print

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<45}{'arguments':<30}")
    ch = [ch]
    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out

    # START layers phinet ------------------------------------
    model = PhiNet(
            input_shape=(3,320,320),
            alpha=2.67,
            num_layers=6,
            beta=1,
            t_zero=4,
            include_top=False,
            num_classes=nc,
            compatibility=False,
        )


    # maybe we can take away this code?
    data_config = {"input_size":(3,320,320)}
    res = get_output_dim(data_config, model)    

    for i, layer in enumerate(model._layers):        
        f = -1
        n = n_ = 1
        args = [2.67, 1, 4] + res[0][i]
        t = str(layer.__class__).replace("<class '",'').replace("'>",'')       
        # get the shape
        
        
        #t = str(m)[8:-2].replace('__main__.', '')  # module type
        layer.np = sum(x.numel() for x in layer.parameters())  # number params
        layer.i, layer.f, layer.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{layer.np:10.0f}  {t:<45}{str(args):<30}')  # print            

        c2 = ch[f]
        if i == 0:
            ch = []

        # da refactorizzare
        try:
            ch.append(res[0][i-1][0])
        except:
            ch.append(c2) # take the number of layers

    layers = list(model._layers)
    # END layers phinet ------------------------------------ 

    for i, (f, n, m, args) in enumerate(d['head'], len(layers)):  # from, number, module, args
        m = getattr(torch.nn, m[3:]) if 'nn.' in m else globals()[m]  # get module
        for j, a in enumerate(args):
            if isinstance(a, str):
                with contextlib.suppress(ValueError):
                    args[j] = locals()[a] if a in locals() else ast.literal_eval(a)

        n = n_ = max(round(n * depth), 1) if n > 1 else n  # depth gain
        if m in (Classify, Conv, ConvTranspose, GhostConv, Bottleneck, GhostBottleneck, SPP, SPPF, DWConv, Focus,
                    BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, nn.ConvTranspose2d, DWConvTranspose2d, C3x, RepC3):
            c1, c2 = ch[f], args[0]
            if c2 != nc:  # if c2 not equal to number of classes (i.e. for Classify() output)
                c2 = make_divisible(min(c2, max_channels) * width, 8)

            args = [c1, c2, *args[1:]]
            if m in (BottleneckCSP, C1, C2, C2f, C3, C3TR, C3Ghost, C3x, RepC3):
                args.insert(2, n)  # number of repeats
                n = 1
        elif m is AIFI:
            args = [ch[f], *args]
        elif m in (HGStem, HGBlock):
            c1, cm, c2 = ch[f], args[0], args[1]
            args = [c1, cm, c2, *args[2:]]
            if m is HGBlock:
                args.insert(4, n)  # number of repeats
                n = 1

        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
        elif m in (Detect, Segment, Pose, RTDETRDecoder):
            args.append([ch[x] for x in f])
            if m is Segment:
                args[2] = make_divisible(min(args[2], max_channels) * width, 8)
        else:
            c2 = ch[f]

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        t = str(m)[8:-2].replace('__main__.', '')  # module type
        m.np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type = i, f, t  # attach index, 'from' index, type
        if verbose:
            LOGGER.info(f'{i:>3}{str(f):>20}{n_:>3}{m.np:10.0f}  {t:<45}{str(args):<30}')  # print
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        
        ch.append(c2)

    # Print the code representation of the layers
    for idx, layer in enumerate(layers[9:]):
        print(f"layer{idx + 1} = {layer}")

def yaml_model_load(path):
    """Load a YOLOv8 model from a YAML file."""
    import re

    path = Path(path)
    if path.stem in (f'yolov{d}{x}6' for x in 'nsmlx' for d in (5, 8)):
        new_stem = re.sub(r'(\d+)([nslmx])6(.+)?$', r'\1\2-p6\3', path.stem)
        LOGGER.warning(f'WARNING ⚠️ Ultralytics YOLO P6 models now use -p6 suffix. Renaming {path.stem} to {new_stem}.')
        path = path.with_stem(new_stem)

    unified_path = re.sub(r'(\d+)([nslmx])(.+)?$', r'\1\3', str(path))  # i.e. yolov8x.yaml -> yolov8.yaml
    yaml_file = check_yaml(unified_path, hard=False) or check_yaml(path)
    d = yaml_load(yaml_file)  # model dict
    d['scale'] = guess_model_scale(path)
    d['yaml_file'] = str(path)
    return d

def guess_model_scale(model_path):
    """
    Takes a path to a YOLO model's YAML file as input and extracts the size character of the model's scale.
    The function uses regular expression matching to find the pattern of the model scale in the YAML file name,
    which is denoted by n, s, m, l, or x. The function returns the size character of the model scale as a string.

    Args:
        model_path (str) or (Path): The path to the YOLO model's YAML file.

    Returns:
        (str): The size character of the model's scale, which can be n, s, m, l, or x.
    """
    with contextlib.suppress(AttributeError):
        import re
        return re.search(r'yolov\d+([nslmx])', Path(model_path).stem).group(1)  # n, s, m, l, or x
    return ''

def init(cfg='yolov8n.yaml', ch=3, nc=None, verbose=True):
    
    yaml = cfg if isinstance(cfg, dict) else yaml_model_load(cfg)  # cfg dict

    # Define model
    ch = yaml['ch'] = yaml.get('ch', ch)  # input channels
    if nc and nc != yaml['nc']:
        LOGGER.info(f"Overriding model.yaml nc={yaml['nc']} with nc={nc}")
        yaml['nc'] = nc  # override yaml value
    parse_model(deepcopy(yaml), ch=ch, verbose=verbose)  # model, savelist

if __name__ == '__main__':
    init()