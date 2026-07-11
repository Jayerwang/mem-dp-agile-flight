import argparse
import os

import torch

from model_memflight import Model


class DeployWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, v, h):
        act, aux, h_next = self.model(x, v, h)
        sigma = aux['sigma']
        return act, sigma, h_next


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--out', default='/root/autodl-tmp/dp_memflight_outputs/exports/memflight.onnx')
    parser.add_argument('--dim_obs', type=int, default=10)
    parser.add_argument('--dim_action', type=int, default=6)
    parser.add_argument('--opset', type=int, default=17)
    args = parser.parse_args()

    model = Model(dim_obs=args.dim_obs, dim_action=args.dim_action)
    state = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(state, strict=False)
    model.eval()

    wrapper = DeployWrapper(model)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    x = torch.randn(1, 1, 12, 16)
    v = torch.randn(1, args.dim_obs)
    h = torch.zeros(1, 256)

    torch.onnx.export(
        wrapper,
        (x, v, h),
        args.out,
        input_names=['x', 'v', 'h'],
        output_names=['act', 'sigma', 'h_next'],
        dynamic_axes={
            'x': {0: 'batch'},
            'v': {0: 'batch'},
            'h': {0: 'batch'},
            'act': {0: 'batch'},
            'sigma': {0: 'batch'},
            'h_next': {0: 'batch'},
        },
        opset_version=args.opset,
    )
    print('exported:', args.out)


if __name__ == '__main__':
    main()
