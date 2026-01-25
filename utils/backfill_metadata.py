import os
import json
import glob
import re
from omegaconf import OmegaConf
from datetime import datetime

EXPERIMENTS_DIR = "experiments"

def update_exp_table():
    table_path = os.path.join('data_log', 'exp_table.md')
    if not os.path.exists(table_path):
        print(f"Table not found: {table_path}")
        return

    with open(table_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    # Parse headers to find column indices
    header_line = None
    header_index = -1
    for i, line in enumerate(lines):
        if '|' in line and 'Model' in line and 'SDR' in line:
            header_line = line
            header_index = i
            break
    
    if header_index == -1:
        print("Could not find table header")
        return

    # Helper to get column index by name
    headers = [h.strip() for h in header_line.split('|')]
    # Remove empty strings from split (start/end)
    headers = [h for h in headers if h]
    
    try:
        idx_plot = headers.index('Plot')
        idx_rec = headers.index('实验记录')
        idx_script = headers.index('实现脚本')
        
        # Metrics indices
        metric_names = ['SDR', 'SISNR', 'PESQ', 'ESTOI', 'STOI']
        metric_indices = {name: headers.index(name) for name in metric_names}
    except ValueError as e:
        print(f"Missing column in table: {e}")
        return

    # Identify rows to update
    targets = []
    for i in range(header_index + 2, len(lines)): # Skip header and separator
        line = lines[i]
        if not line.strip() or '|' not in line:
            continue
        
        cols = [c.strip() for c in line.split('|')]
        # Handle the split resulting in empty first/last elements
        if not cols[0]: cols = cols[1:]
        if not cols[-1]: cols = cols[:-1]
        
        if len(cols) <= max(idx_plot, idx_rec, idx_script):
             continue

        # Extract metrics for comparison
        row_metrics = {}
        try:
            for name, idx in metric_indices.items():
                val = cols[idx]
                if val and val != '-':
                    row_metrics[name] = float(val)
        except ValueError:
            continue # Skip if metrics aren't parseable
        
        if row_metrics:
            targets.append({
                'line_index': i,
                'metrics': row_metrics
            })

    if not targets:
        print("No target rows found in exp_table.md.")
        return

    # Traverse experiments
    exp_dir = EXPERIMENTS_DIR
    if not os.path.exists(exp_dir):
        print(f"Experiments directory {exp_dir} not found.")
        return

    for exp_name in os.listdir(exp_dir):
        exp_path = os.path.join(exp_dir, exp_name)
        if not os.path.isdir(exp_path):
            continue

        # Check for train scripts
        has_train = os.path.exists(os.path.join(exp_path, 'train.py'))
        has_train2 = os.path.exists(os.path.join(exp_path, 'train2.py'))

        if has_train and has_train2:
            continue # Skip exception - both exist
        
        script_file = 'train.py' if has_train else ('train2.py' if has_train2 else None)
        if not script_file:
            continue

        # Find RESULTS.txt
        # Pattern: best_model_*/enhanced_dns/scoring_intrusive/RESULTS.txt
        results_glob = os.path.join(exp_path, 'best_model_*', 'enhanced_dns', 'scoring_intrusive', 'RESULTS.txt')
        result_files = glob.glob(results_glob)
        
        if not result_files:
            continue
        
        # Assume first match is the one
        result_file = result_files[0]
        
        exp_metrics = {}
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if ':' in line:
                        key, val = line.split(':', 1)
                        key = key.strip()
                        if key in metric_names:
                            exp_metrics[key] = float(val.strip())
        except Exception:
            continue
        
        # Compare with targets
        for target in targets:
            match = True
            if not exp_metrics: 
                match = False
            
            for m, val in target['metrics'].items():
                if m in exp_metrics:
                   if abs(exp_metrics[m] - val) > 1e-3: # Tolerance
                       match = False
                       break
                else:
                    pass
            
            if match:
                # Extract model name
                model_module = ""
                try:
                    with open(os.path.join(exp_path, script_file), 'r', encoding='utf-8') as f:
                        content = f.read()
                        # from models.gtcrn_dynamic_per_band_acrean import GTCRN as Model
                        match_import = re.search(r'from models\.(\S+)\s+import\s+\w+\s+as\s+Model', content)
                        if match_import:
                            model_module = match_import.group(1)
                except Exception as e:
                    print(f"Error reading {script_file} in {exp_name}: {e}")

                # Update the target line
                line_idx = target['line_index']
                line = lines[line_idx]
                parts = line.split('|')
                
                # Timestamp
                # exp_gtcrn_2025-11-05-17h31m -> 2025-11-05-17h31m
                ts_match = re.search(r'(\d{4}-\d{2}-\d{2}-\d{2}h\d{2}m)', exp_name)
                timestamp = ts_match.group(1) if ts_match else exp_name

                # Update implementation script (idx_script + 1 because of split)
                if model_module:
                    parts[idx_script + 1] = f" {model_module} "
                
                # Update experiment record
                parts[idx_rec + 1] = f" {timestamp} "
                
                lines[line_idx] = "|".join(parts)
                print(f"Matched {exp_name} to table row at line {line_idx}. Updated.")

    with open(table_path, 'w', encoding='utf-8') as f:
        f.writelines(lines)
    print("Table update completed.")

def backfill():
    if not os.path.exists(EXPERIMENTS_DIR):
        print(f"Directory {EXPERIMENTS_DIR} does not exist.")
        return

    for exp_name in os.listdir(EXPERIMENTS_DIR):
        exp_path = os.path.join(EXPERIMENTS_DIR, exp_name)
        if not os.path.isdir(exp_path):
            continue
            
        meta_path = os.path.join(exp_path, 'metadata.json')
        if os.path.exists(meta_path):
            print(f"Skipping {exp_name} (metadata exists)")
            continue
            
        print(f"Processing {exp_name}...")
        
        # Try to load config
        config_path = os.path.join(exp_path, 'config.yaml')
        config_data = {}
        if os.path.exists(config_path):
            try:
                conf = OmegaConf.load(config_path)
                config_data = OmegaConf.to_container(conf, resolve=True)
            except:
                print(f"  - Failed to load config.yaml")

        # Find best checkpoint
        best_epoch = None
        best_score = None
        checkpoints = glob.glob(os.path.join(exp_path, 'checkpoints', 'best_model_*.tar'))
        if checkpoints:
            try:
                # filename format: best_model_005.tar or similar?
                # train.py: 'best_model_{}.tar'.format(str(self.state_dict_best['epoch']).zfill(3))
                # So we can extract epoch
                fname = os.path.basename(checkpoints[0])
                # best_model_123.tar
                epoch_str = fname.replace('best_model_', '').replace('.tar', '')
                best_epoch = int(epoch_str)
            except:
                pass
        
        # Heuristic for timestamp from folder name
        # exp_gtcrn_2026-01-17-15h50m
        timestamp = "?"
        try:
            parts = exp_name.split('_')
            # Assuming last part is time and second last is date
            # But the folder name format in train.py is:
            # base_exp_path + '_' + datetime.now().strftime("%Y-%m-%d-%Hh%Mm")
            # So everything after the last underscore is datetime? 
            # Wait, timestamp in format %Y-%m-%d-%Hh%Mm has dashes.
            # "exp_gtcrn_2026-01-17-15h50m" -> "2026-01-17-15h50m"
            ts_str = parts[-1] 
            # Simple check if it looks like a date
            if '-' in ts_str and 'h' in ts_str:
                timestamp = ts_str
        except:
             pass

        metadata = {
            "exp_id": exp_name,
            "exp_path": os.path.abspath(exp_path),
            "timestamp": timestamp,
            "status": "completed" if best_epoch else "unknown",
            "model_path": "unknown (legacy)",
            "model_classname": "unknown (legacy)",
            "run_config": config_data,
            "best_epoch": best_epoch
        }
        
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"  - Created metadata.json")

if __name__ == "__main__":
    # backfill()
    update_exp_table()
