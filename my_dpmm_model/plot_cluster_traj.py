import os
import json
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

def visualize_waypoints(json_file_path, timesteps=np.arange(-10,10)):
    """
    Visualize waypoints from a specific JSON file with 1-sigma uncertainty ellipses.
    
    Parameters:
    json_file_path (str): Exact path to the JSON file to visualize
    """
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return
    
    # Read the JSON data
    with open(json_file_path, 'r') as f:
        components = json.load(f)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Process each component (k)
    for component in components:
        k = component['k']
        mu = np.array(component['mu'])
        var = np.array(component['var'])
        
        # Extract x and y coordinates (even indices are x, odd are y)
        x_coords = mu[::2]  # x coordinates
        y_coords = mu[1::2]  # y coordinates
        x_var = var[::2]    # x variances
        y_var = var[1::2]   # y variances
        
        # Calculate 1-sigma standard deviations
        x_sigma = np.sqrt(x_var)
        y_sigma = np.sqrt(y_var)
        
        # Create timestep labels (from -5 to 4)
        # timesteps = np.arange(-5, 5)
        
        # Plot the waypoints trajectory
        line, = plt.plot(x_coords, y_coords, marker='o', label=f'Component {k}', linewidth=2)
        color = line.get_color()
        
        # Add 1-sigma uncertainty ellipses at each waypoint
        for i, (x, y, x_sig, y_sig) in enumerate(zip(x_coords, y_coords, x_sigma, y_sigma)):
            # Create ellipse for 1-sigma confidence region
            ellipse = Ellipse((x, y), width=2*x_sig, height=2*y_sig, 
                            fill=False, color=color, alpha=0.5, linestyle='--')
            plt.gca().add_patch(ellipse)
        
        # Add timestep annotations (only for first few points to avoid clutter)
        for i, (x, y, t) in enumerate(zip(x_coords, y_coords, timesteps)):
            if i % 2 == 0 or i == len(timesteps) - 1:  # Show every other annotation + last point
                plt.annotate(f't={t}', (x, y), xytext=(10, 10), 
                            textcoords='offset points', fontsize=7, alpha=0.8,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Waypoints Visualization with 1-Sigma Uncertainty - {os.path.basename(json_file_path)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()

# Alternative version with error bars instead of ellipses
def visualize_waypoints_errorbars(json_file_path, timesteps=np.arange(-10,10)):
    """
    Visualize waypoints with error bars showing 1-sigma uncertainty.
    
    Parameters:
    json_file_path (str): Exact path to the JSON file to visualize
    """
    
    # Check if file exists
    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return
    
    # Read the JSON data
    with open(json_file_path, 'r') as f:
        components = json.load(f)
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Process each component (k)
    for component in components:
        k = component['k']
        mu = np.array(component['mu'])
        var = np.array(component['var'])
        
        # Extract x and y coordinates
        x_coords = mu[::2]  # x coordinates
        y_coords = mu[1::2]  # y coordinates
        x_var = var[::2]    # x variances
        y_var = var[1::2]   # y variances
        
        # Calculate 1-sigma standard deviations
        x_sigma = np.sqrt(x_var)
        y_sigma = np.sqrt(y_var)
        
        # Create timestep labels
        # timesteps = np.arange(-5, 5)
        
        # Plot the waypoints with error bars
        plt.errorbar(x_coords, y_coords, xerr=x_sigma, yerr=y_sigma,
                    marker='o', label=f'Component {k}', linewidth=2,
                    capsize=3, capthick=1, alpha=0.8)
        
        # Also plot the trajectory line
        plt.plot(x_coords, y_coords, linewidth=1, alpha=0.5)
        
        # Add timestep annotations
        for i, (x, y, t) in enumerate(zip(x_coords, y_coords, timesteps)):
            if i % 3 == 0 or i == len(timesteps) - 1:  # Show fewer annotations to avoid clutter
                plt.annotate(f't={t}', (x, y), xytext=(8, 8), 
                            textcoords='offset points', fontsize=7, alpha=0.7,
                            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6))
    
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'Waypoints Visualization with 1-Sigma Error Bars - {os.path.basename(json_file_path)}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    plt.tight_layout()
    plt.show()


def visualize_all_waypoints(date_dir, output_dir=None):
    """
    Visualize waypoints from all JSON files in a date directory and save plots.
    
    Parameters:
    date_dir (str): Path to the date directory containing component_log folder
    output_dir (str): Path to save plots (default: same as date_dir/plots)
    """
    
    # Define the component_log directory
    component_log_dir = os.path.join(date_dir, 'component_log')
    
    # Set output directory
    if output_dir is None:
        output_dir = os.path.join(date_dir, 'cluster_traj_plots')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if component_log directory exists
    if not os.path.exists(component_log_dir):
        print(f"Component log directory not found: {component_log_dir}")
        return
    
    # Find all JSON files
    json_files = []
    for file in os.listdir(component_log_dir):
        if file.endswith('-components.json'):
            json_files.append(os.path.join(component_log_dir, file))
    
    if not json_files:
        print(f"No JSON files found in {component_log_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process...")
    
    # Process each JSON file
    for i, json_file_path in enumerate(json_files):
        print(f"Processing {i+1}/{len(json_files)}: {os.path.basename(json_file_path)}")
        
        try:
            # Read the JSON data
            with open(json_file_path, 'r') as f:
                components = json.load(f)
            
            # Create the plot
            plt.figure(figsize=(12, 8))
            
            # Process each component (k)
            for component in components:
                k = component['k']
                mu = np.array(component['mu'])
                var = np.array(component['var'])
                
                # Extract x and y coordinates
                end_point = 2*9
                x_coords = mu[0:end_point:2]  # x coordinates
                y_coords = mu[1:end_point:2]  # y coordinates
                x_var = var[0:end_point:2]    # x variances
                y_var = var[1:end_point:2]   # y variances
                
                # Calculate 1-sigma standard deviations
                x_sigma = np.sqrt(x_var)
                y_sigma = np.sqrt(y_var)
                
                # Plot the waypoints trajectory
                line, = plt.plot(x_coords, y_coords, marker='o', label=f'Component {k}', linewidth=2)
                color = line.get_color()
                
                # Add 1-sigma uncertainty ellipses at each waypoint
                for j, (x, y, x_sig, y_sig) in enumerate(zip(x_coords, y_coords, x_sigma, y_sigma)):
                    # Create ellipse for 1-sigma confidence region
                    ellipse = Ellipse((x, y), width=2*x_sig, height=2*y_sig, 
                                    fill=False, color=color, alpha=0.5, linestyle='--')
                    plt.gca().add_patch(ellipse)
                
                # Add timestep annotations (reduce clutter)
                # timesteps = np.arange(-5, 5)
                # for j, (x, y, t) in enumerate(zip(x_coords, y_coords, timesteps)):
                #     if j % 3 == 0 or j == len(timesteps) - 1:  # Show every 3rd annotation + last point
                #         plt.annotate(f't={t}', (x, y), xytext=(10, 10), 
                #                     textcoords='offset points', fontsize=7, alpha=0.8,
                #                     bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
            
            # Configure plot
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            plt.title(f'Waypoints Visualization - {os.path.basename(json_file_path)}')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.axis('equal')
            plt.tight_layout()
            
            # Save plot with same name as JSON file but .png extension
            json_filename = os.path.basename(json_file_path)
            plot_filename = json_filename.replace('-components.json', '.png')
            plot_filepath = os.path.join(output_dir, plot_filename)
            
            plt.savefig(plot_filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Close figure to free memory
            
            print(f"Saved plot: {plot_filename}")
            
        except Exception as e:
            print(f"Error processing {json_file_path}: {str(e)}")
            plt.close()  # Make sure to close figure even if error occurs
    
    print(f"All plots saved to: {output_dir}")

if __name__=='__main__':
    # Example usage:
    # visualize_waypoints('./results/2025-07-25-20-37/component_log/7-0-1114-components.json')
    # visualize_waypoints_errorbars('./results/2025-07-25-20-37/component_log/7-0-1114-components.json')
    # visualize_waypoints('./results/2025-07-25-22-13/component_log/7-0-1144-components.json', timesteps=np.arange(-5,5))
    # visualize_all_waypoints(date_dir='./results/2025-07-25-20-37')
    visualize_all_waypoints(date_dir='./results/2025-07-25-22-13',timesteps=np.arange(-5,5))