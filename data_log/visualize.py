import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np
import yaml

def parse_table(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    header = [h.strip() for h in lines[0].strip().strip('|').split('|')]
    data = []
    for line in lines[2:]:
        if not line.strip():
            continue
        
        # Clean the line for parsing
        cleaned_line = line.replace('**', '').strip().strip('|')
        values = [v.strip() for v in cleaned_line.split('|')]
        
        # Extract model index from the first column
        model_name = values[0]
        match = re.match(r'(\d+)', model_name)
        if match:
            model_index = int(match.group(1))
        else:
            # Fallback if no number is at the start
            model_index = len(data) + 1

        row_data = {'index': model_index}
        
        for i, col_name in enumerate(header):
            # The first column is 'Model', which we've handled
            if i == 0:
                row_data[col_name] = model_name
                continue

            try:
                value_str = values[i]
            except IndexError:
                value_str = '' # Handle rows with missing columns

            if col_name == 'Plot':
                row_data[col_name] = value_str
                continue
            
            # Handle special cases like '0.036+-0.005647' for time and '(34.73)' for computation
            if '+-' in value_str:
                value_str = value_str.split('+-')[0]
            
            value_str = value_str.replace('(', '').replace(')', '')

            try:
                row_data[col_name] = float(value_str)
            except (ValueError, IndexError):
                row_data[col_name] = np.nan
        
        data.append(row_data)
        
    return pd.DataFrame(data)

def load_plot_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config.get('color_map', {}), config.get('legend_labels', {})

def plot_data(df, x_col, y_col, title, x_label, y_label, output_path, color_map, legend_labels, fit_curve=False):
    fig, ax = plt.subplots(figsize=(12, 8))

    # Filter out rows where 'Plot' is empty
    plot_df = df[df['Plot'].notna() & (df['Plot'] != '')].copy()

    plot_df['color_code'] = plot_df['Plot']
    plot_df['color'] = plot_df['Plot'].map(color_map).fillna('white')

    # Plot points for each color group to create legend entries
    for color_code, label in legend_labels.items():
        color_name = color_map.get(color_code, 'white')
        subset = plot_df[plot_df['color_code'] == color_code]
        if not subset.empty:
            ax.scatter(subset[x_col], subset[y_col], c=color_name, s=100, zorder=5, label=label)

            if fit_curve and len(subset[x_col].dropna()) > 1:
                # Sort values for a smooth line plot
                sorted_subset = subset.dropna(subset=[x_col, y_col]).sort_values(by=x_col)
                x_fit = sorted_subset[x_col]
                y_fit = sorted_subset[y_col]
                
                # Perform linear regression
                try:
                    # Fit a 2nd degree polynomial
                    coeffs = np.polyfit(x_fit, y_fit, 2)
                    p = np.poly1d(coeffs)
                    
                    x_line = np.linspace(x_fit.min(), x_fit.max(), 100)
                    y_line = p(x_line)
                    
                    ax.plot(x_line, y_line, color=color_name, linestyle='--', linewidth=2, zorder=6)
                except (np.linalg.LinAlgError, ValueError):
                    print(f"Could not fit curve for {label}, not enough data points or other error.")

    # Annotate points with their index
    for _, row in plot_df.iterrows():
        if pd.notna(row[x_col]) and pd.notna(row[y_col]):
            ax.text(row[x_col], row[y_col], str(row['index']), color='white', 
                    ha='center', va='center', fontsize=9, zorder=10)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    
    # Add legend
    ax.legend(title="Model Types", fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved plot to {output_path}")

if __name__ == "__main__":
    # Create figures directory if it doesn't exist
    if not os.path.exists('figures'):
        os.makedirs('figures')

    # Load and parse data
    df = parse_table('exp_table.md')

    # Load plot configuration
    color_map, legend_labels = load_plot_config('plot_config.yaml')

    # Plot 1: PESQ vs CPU inference time
    plot_data(df, 
              x_col='CPU推理时间', 
              y_col='PESQ', 
              title='PESQ vs. CPU inference time', 
              x_label='CPU inference time (s)', 
              y_label='PESQ', 
              output_path='figures/pesq_vs_inference_time.png',
              color_map=color_map,
              legend_labels=legend_labels,
              fit_curve=False)

    # Plot 2: PESQ vs CPU computation
    plot_data(df, 
              x_col='CPU计算量', 
              y_col='PESQ', 
              title='PESQ vs. CPU computation', 
              x_label='CPU computation', 
              y_label='PESQ', 
              output_path='figures/pesq_vs_computation.png',
              color_map=color_map,
              legend_labels=legend_labels,
              fit_curve=False)
    
    # Plot 3: P808_MOS vs CPU computation
    plot_data(df,
              x_col='CPU计算量',
              y_col='P808_MOS',
              title='P808_MOS vs. CPU computation',
              x_label='CPU computation',
              y_label='P808_MOS',
              output_path='figures/p808_mos_vs_computation.png',
              color_map=color_map,
              legend_labels=legend_labels,
              fit_curve=False)
    
    # Plot 4: SDR vs CPU computation
    plot_data(df,
              x_col='CPU计算量',
              y_col='SDR',
              title='SDR vs. CPU computation',
              x_label='CPU computation',
              y_label='SDR',
              output_path='figures/sdr_vs_computation.png',
              color_map=color_map,
              legend_labels=legend_labels,
              fit_curve=False)