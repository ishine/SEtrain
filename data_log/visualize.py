import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import numpy as np

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

def plot_data(df, x_col, y_col, title, x_label, y_label, output_path):
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 8))

    # Filter out rows where 'Plot' is empty
    plot_df = df[df['Plot'].notna() & (df['Plot'] != '')].copy()

    # Define a color mapping if you want to map short codes to color names
    color_map = {
        'r': 'red',
        'g': 'green',
        'b': 'blue',
        'y': 'yellow',
        'c': 'cyan',
        'm': 'magenta'
    }
    plot_df['color'] = plot_df['Plot'].map(color_map).fillna('white')

    # Plot points
    ax.scatter(plot_df[x_col], plot_df[y_col], c=plot_df['color'], s=100, zorder=5)

    # Annotate points with their index
    for _, row in plot_df.iterrows():
        if pd.notna(row[x_col]) and pd.notna(row[y_col]):
            ax.text(row[x_col], row[y_col], str(row['index']), color='white', 
                    ha='center', va='center', fontsize=12, zorder=10)

    ax.set_xlabel(x_label, fontsize=16)
    ax.set_ylabel(y_label, fontsize=16)
    ax.set_title(title, fontsize=20)
    ax.grid(True, which='both', linestyle='--', linewidth=0.5, color='gray')
    
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

    # Plot 1: PESQ vs CPU inference time
    plot_data(df, 
              x_col='CPU推理时间', 
              y_col='PESQ', 
              title='PESQ vs. CPU inference time', 
              x_label='CPU inference time (s)', 
              y_label='PESQ', 
              output_path='figures/pesq_vs_inference_time.png')

    # Plot 2: PESQ vs CPU computation
    plot_data(df, 
              x_col='CPU计算量', 
              y_col='PESQ', 
              title='PESQ vs. CPU computation', 
              x_label='CPU computation', 
              y_label='PESQ', 
              output_path='figures/pesq_vs_computation.png')
    
    # Plot 3: P808_MOS vs CPU computation
    plot_data(df,
              x_col='CPU计算量',
              y_col='P808_MOS',
              title='P808_MOS vs. CPU computation',
              x_label='CPU computation',
              y_label='P808_MOS',
              output_path='figures/p808_mos_vs_computation.png')
