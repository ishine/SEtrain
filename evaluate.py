import os
from omegaconf import OmegaConf
import sys


def main(args):
    config = OmegaConf.load(args.config)
    enh_folder = config.network.enh_folder
    # enh_folder = '/data/ssd0/xiaobin.rong/Datasets/DNS3/test_noisy/'
    
    if args.metric == 'dnsmos' or args.metric == 'all':
        os.system(
            (f'{sys.executable} ./evaluation/calculate_nonintrusive_dnsmos.py '
                f'--inf_scp {enh_folder}/inf.scp '
                f'--output_dir {enh_folder}/scoring_dnsmos '
                f'--device {args.dnsmos_device} '
                '--job 1 '
                '--convert_to_torch False '
                '--primary_model ./DNSMOS/DNSMOS/sig_bak_ovr.onnx '
                '--p808_model ./DNSMOS/DNSMOS/model_v8.onnx'
            )
        )    
    if args.metric == 'intrusive' or args.metric == 'all':
        os.system(
            (f'{sys.executable} ./evaluation/calculate_intrusive_se_metrics.py '
             f'--ref_scp {enh_folder}/ref.scp '
             f'--inf_scp {enh_folder}/inf.scp '
             f'--output_dir {enh_folder}/scoring_intrusive '
             '--nj 8 '
             '--chunksize 1000'
            )
        )

    else:
        raise ValueError
    

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--metric', default='all', help="Metric to be calculated")
    parser.add_argument('--config', default='configs/cfg_infer.yaml')
    parser.add_argument('--device', default='0')
    parser.add_argument('--dnsmos_device', default='cpu', help="Device for DNSMOS evaluation: 'cpu' or 'cuda'")
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    main(args)
