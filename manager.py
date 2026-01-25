import argparse
import subprocess
import os
import sys
import json
import glob
import re
import time
from datetime import datetime
from omegaconf import OmegaConf

def print_step(msg):
    print(f"\n\033[94m[MANAGER Step] {msg}\033[0m")

def print_success(msg):
    print(f"\033[92m{msg}\033[0m")

def print_error(msg):
    print(f"\033[91m{msg}\033[0m")

def start(args):
    print_step("Preparing Training Configuration...")
    base_config = OmegaConf.load(args.config)
    
    if args.overrides:
        print(f"Applying overrides: {args.overrides}")
        override_conf = OmegaConf.from_dotlist(args.overrides)
        base_config = OmegaConf.merge(base_config, override_conf)
    
    temp_train_config = "temp_train_config.yaml"
    OmegaConf.save(base_config, temp_train_config)
    
    print_step(f"Starting Training with {temp_train_config} on GPU {args.gpu}...")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    
    cmd = [sys.executable, "train_auto.py", "-C", temp_train_config, "-D", args.gpu]
    
    exp_path = None
    
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env) as proc:
        for line in proc.stdout:
            print(line, end='')
            if "EXP_PATH_GENERATED:" in line:
                exp_path = line.strip().split("EXP_PATH_GENERATED:")[1]
    
    if proc.returncode != 0:
        print_error("Training failed.")
        return

    if not exp_path:
        print_error("Could not capture experiment path from training output.")
        return
        
    print_success(f"Training finished. Experiment Path: {exp_path}")

    print_step("Preparing Inference Configuration...")
    
    metadata_path = os.path.join(exp_path, 'metadata.json')
    if not os.path.exists(metadata_path):
        print_error(f"Metadata file not found at {metadata_path}")
        return
        
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    metadata["exp_name"] = args.name or "Unnamed Experiment"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)
        
    best_epoch = metadata.get('best_epoch')
    if best_epoch is None:
        # Fallback: look for file
        checkpoints = glob.glob(os.path.join(exp_path, 'checkpoints', 'best_model_*.tar'))
        if not checkpoints:
            print_error("No best model checkpoint found.")
            return
        ckpt_filename = os.path.basename(checkpoints[0])
        ckpt_name = ckpt_filename.replace('.tar', '')
    else:
        ckpt_name = f"best_model_{str(best_epoch).zfill(3)}"
    
    print(f"Using checkpoint: {ckpt_name}")
    
    # Create inference config content
    infer_config_dict = {
        "test_dataset": {
            "noisy_dir": "/home/nis/zhiheng.wang/SEtrain/DNS3/test_noisy",
            "clean_dir": "/home/nis/zhiheng.wang/SEtrain/DNS3/test_clean"
        },
        "network": {
            "exp_path": exp_path,
            "config": f"{exp_path}/config.yaml",
            "ckpt_name": ckpt_name,
            "checkpoint": f"{exp_path}/checkpoints/{ckpt_name}.tar",
            "enh_folder": f"{exp_path}/{ckpt_name}/enhanced_dns"
        },
        "model": {
            "path": metadata['model_path'],
            "classname": metadata['model_classname']
        }
    }
    
    infer_conf = OmegaConf.create(infer_config_dict)
    temp_infer_config = "temp_infer_config.yaml"
    OmegaConf.save(infer_conf, temp_infer_config)
    
    print_step("Starting Inference...")
    cmd_infer = [sys.executable, "infer_auto.py", "-C", temp_infer_config]
    subprocess.run(cmd_infer, check=True, env=env)
    
    print_step("Starting Evaluation...")
    # We can reuse temp_infer_config
    cmd_eval = [sys.executable, "evaluate.py", "--config", temp_infer_config, "--metric", "all", "--device", args.gpu]
    subprocess.run(cmd_eval, check=True, env=env)
    
    print_step("Parsing Results...")
    enh_folder = infer_config_dict['network']['enh_folder']
    
    metrics = {
        "SDR": "-", "SISNR": "-", "PESQ": "-", "ESTOI": "-", "STOI": "-", 
        "OVRL": "-", "SIG": "-", "BAK": "-", "P808_MOS": "-"
    }

    def parse_results_file(file_path):
        parsed = {}
        if os.path.exists(file_path):
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                    for line in content.splitlines():
                        if ":" in line:
                            parts = line.split(':')
                            key = parts[0].strip()
                            try:
                                val_str = parts[1].strip().split()[0]
                                val = float(val_str)
                                parsed[key] = val
                            except:
                                pass
            except Exception as e:
                print_error(f"Error reading {file_path}: {e}")
        else:
            print(f"File not found: {file_path}")
        return parsed

    intrusive_file = os.path.join(enh_folder, "scoring_intrusive", "RESULTS.txt")
    intrusive_data = parse_results_file(intrusive_file)
    for k in ["SDR", "SISNR", "PESQ", "ESTOI", "STOI"]:
        val = intrusive_data.get(k) or intrusive_data.get(k.lower()) or intrusive_data.get(k.upper())
        if val is not None:
            metrics[k] = f"{val:.4f}"

    dnsmos_file = os.path.join(enh_folder, "scoring_dnsmos", "RESULTS.txt")
    dnsmos_data = parse_results_file(dnsmos_file)
    
    mapping = {
        "OVRL": ["OVRL", "ovrl", "Overall"],
        "SIG": ["SIG", "sig", "Signal"],
        "BAK": ["BAK", "bak", "Background"],
        "P808_MOS": ["P808_MOS", "p808_mos", "P808"]
    }
    
    for target_key, search_keys in mapping.items():
        for sk in search_keys:
            if sk in dnsmos_data:
                metrics[target_key] = f"{dnsmos_data[sk]:.4f}"
                break

    print("Parsed Metrics:", metrics)

    try:
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            metadata['metrics'] = metrics
            metadata['status'] = "completed"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=4)
    except Exception as e:
        print_error(f"Failed to update metadata with metrics: {e}")

    try:
        table_path = "data_log/exp_table.md"
        # row_id = f"**{metadata.get('exp_id', 'New Run')}**" 
        model_name = args.name or metadata.get('model_path', 'Unknown')
        
        timestamp = metadata.get('timestamp')
        if timestamp:
            try:
                timestamp = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S").strftime("%Y-%m-%d-%Hh%Mm")
            except:
                pass
        else:
            timestamp = datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
        
        # Determine params/macs from log file if possible, else placeholders
        # Assuming placeholders for now as calculating them requires parsing log output or model introspection
        params = "-"
        macs = "-"
        inference_time = "-"
        
        # Construct Row
        # | Model | Plot | SDR | SISNR | PESQ | ESTOI | STOI | OVRL | SIG | BAK | P808_MOS | 参数量 | CPU计算量 | CPU推理时间 | 实现脚本 | 实验记录 |
        
        row = f"| **{model_name} (Auto)** | g | {metrics['SDR']} | {metrics['SISNR']} | {metrics['PESQ']} | {metrics['ESTOI']} | {metrics['STOI']} | {metrics['OVRL']} | {metrics['SIG']} | {metrics['BAK']} | {metrics['P808_MOS']} | {params} | {macs} | {inference_time} | {metadata.get('model_path', 'auto')} | {timestamp} |"
        
        if os.path.exists(table_path):
            with open(table_path, 'a') as f:
                f.write(f"\n{row}")
            print_success(f"Added results to {table_path}")
        else:
            print_error(f"{table_path} not found.")

    except Exception as e:
         print_error(f"Failed to update exp_table.md: {e}")

    print_success("Pipeline completed!")


def list_experiments(args):
    root_dir = "experiments"
    experiments = []
    
    if not os.path.exists(root_dir):
        print("No experiments found.")
        return

    print(f"{'ID':<40} {'Model':<25} {'Created':<20} {'Status':<15} {'Best Epoch'}")
    print("-" * 115)
    
    for exp_name in sorted(os.listdir(root_dir), reverse=True):
        exp_path = os.path.join(root_dir, exp_name)
        meta_path = os.path.join(exp_path, 'metadata.json')
        
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
                
                exp_id = meta.get('exp_id', exp_name)
                model = f"{meta.get('model_classname', '?')} ({meta.get('model_path', '?').split('.')[-1]})"
                date = meta.get('timestamp', '?')
                status = meta.get('status', '?')
                best = meta.get('best_epoch', '-')
                
                print(f"{exp_id:<40} {model:<25} {date:<20} {status:<15} {best}")
            except:
                pass

def board(args):
    root_dir = "experiments"
    logdirs = []
    
    labels = args.ids
    
    # If args.ids is empty, list top 5 recent
    targets = []
    if not labels:
        all_exps = sorted(os.listdir(root_dir), reverse=True)
        targets = all_exps[:5]
    else:
        for label in labels:
            matched = False
            for exp in os.listdir(root_dir):
                if label in exp:
                    targets.append(exp)
                    matched = True
                    break # Match first
                metadata_file = os.path.join(root_dir, exp, 'metadata.json')
                with open(metadata_file, 'r') as f:
                    meta = json.load(f)
                exp_name = meta.get('exp_name', '') or meta.get('model_path', '')
                if label in exp_name:
                    targets.append(exp)
                    matched = True
                    break
            if not matched:
                print(f"Warning: No experiment found matching '{label}'")

    if not targets:
        print("No experiments to visualize.")
        return

    # Construct logdir string
    # format: name1:/path/to/exp1/logs,name2:/path/to/exp2/logs
    specs = []
    for t in targets:
        path = os.path.join(os.path.abspath(root_dir), t, 'logs')
        metadata_file = os.path.join(root_dir, t, 'metadata.json')
        with open(metadata_file, 'r') as f:
            meta = json.load(f)
        exp_name = meta.get('exp_name', '') or meta.get('model_path', '')
        # short name for label
        # e.g. exp_gtcrn_2026... -> gtcrn_15h50
        name_parts = t.split('_')
        short_name = f"{name_parts[1]}_{name_parts[-1][-5:]}" if len(name_parts) > 2 else t
        specs.append(f"{exp_name or short_name}:{path}")
    
    logdir_spec = ",".join(specs)
    print(f"Starting TensorBoard with {len(targets)} experiments...")
    
    cmd = ["tensorboard", "--logdir_spec", logdir_spec]
    print(f"Run: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(description="SE Experiment Manager")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Start Command
    start_parser = subparsers.add_parser("start", help="Start a new training pipeline")
    start_parser.add_argument("-C", "--config", default="configs/cfg_train_auto.yaml", help="Base training config")
    start_parser.add_argument("-g", "--gpu", default="0", help="GPU index")
    start_parser.add_argument("overrides", nargs="*", help="Config overrides (e.g. model.classname=Mamba2)")
    start_parser.add_argument("--name", default=None, help="Optional experiment name")
    
    # List Command
    list_parser = subparsers.add_parser("list", help="List experiments")
    
    # Board Command
    board_parser = subparsers.add_parser("board", help="Start TensorBoard for specific experiments")
    board_parser.add_argument("ids", nargs="*", help="Experiment IDs or partial strings to match")
    
    args = parser.parse_args()
    
    if args.command == "start":
        start(args)
    elif args.command == "list":
        list_experiments(args)
    elif args.command == "board":
        board(args)

if __name__ == "__main__":
    main()
