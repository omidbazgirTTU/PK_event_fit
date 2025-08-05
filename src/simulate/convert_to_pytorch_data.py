import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def convert_npz_to_pytorch_csv():
    """
    Convert existing pk_dataset_2C.npz to CSV format expected by PyTorch implementation.
    This ensures consistency with the existing 2C_simulator.py parameters.
    """
    
    # Load the existing NPZ data from data directory
    data_path = "../../data/pk_dataset_2C.npz"
    try:
        data = np.load(data_path)
        all_solutions = data["all_solutions"]
        print(f"Loaded pk_dataset_2C.npz with shape: {all_solutions.shape}")
    except FileNotFoundError:
        print(f"pk_dataset_2C.npz not found at {data_path}")
        print("Please run 2C_simulator.py first to generate the data.")
        return None, None
    
    # Extract parameters from 2C_simulator.py to maintain consistency
    dose_times = [12.0, 24.0, 36.0]  # hours (from 2C_simulator.py)
    dose_amount = 100.0              # mg (from 2C_simulator.py)
    t_final = 48.0                   # hours (from 2C_simulator.py)
    
    # Use first patient's data for PyTorch training (index 0)
    patient_data = all_solutions[0]  # shape: (4, 200, 2)
    
    # Reconstruct time points from the segmented data
    n_segments, steps_per_segment, _ = patient_data.shape
    segment_bounds = [0.0] + dose_times + [t_final]
    
    all_times = []
    all_concentrations = []  # Central compartment (amounts, not concentrations)
    
    for seg_idx in range(n_segments):
        t_start = segment_bounds[seg_idx]
        t_end = segment_bounds[seg_idx + 1]
        
        # Time points for this segment
        segment_times = np.linspace(t_start, t_end, steps_per_segment)
        
        # Central compartment amounts for this segment
        segment_concentrations = patient_data[seg_idx, :, 0]  # Central compartment
        
        all_times.extend(segment_times)
        all_concentrations.extend(segment_concentrations)
    
    all_times = np.array(all_times)
    all_concentrations = np.array(all_concentrations)
    
    # Convert to concentrations (divide by volume)
    # From 2C_simulator.py: mean_params = [CL, V1, Q, V2] = [1.0, 10.0, 0.5, 20.0]
    V1 = 10.0  # Central volume from 2C_simulator.py
    true_concentrations = all_concentrations / V1  # Convert amounts to concentrations
    
    # Add realistic noise (5% CV)
    np.random.seed(42)  # For reproducibility
    noise_level = 0.05
    noise = np.random.normal(0, noise_level * true_concentrations, len(true_concentrations))
    observed_concentrations = true_concentrations + noise
    
    # Ensure no negative concentrations
    observed_concentrations = np.maximum(observed_concentrations, 0.0)
    
    # Create DataFrames with consistent format
    pk_data = pd.DataFrame({
        'Day': all_times / 24.0,  # Convert hours to days for PyTorch consistency
        'True_Concentration (mg/L)': true_concentrations,
        'Concentration_with_noise (mg/L)': observed_concentrations
    })
    
    dosing_schedule = pd.DataFrame({
        'Day': np.array(dose_times) / 24.0,  # Convert to days
        'Dose_Amount (mg)': [dose_amount] * len(dose_times)  # All doses are same amount
    })
    
    # Save to data directory
    data_dir = "../../data"
    
    pk_data.to_csv(os.path.join(data_dir, 'pk_simulation_data.csv'), index=False)
    dosing_schedule.to_csv(os.path.join(data_dir, 'dosing_schedule.csv'), index=False)
    
    print(f"\nGenerated PyTorch-compatible data:")
    print(f"- Time range: {pk_data['Day'].min():.2f} to {pk_data['Day'].max():.2f} days")
    print(f"- Number of observations: {len(pk_data)}")
    print(f"- Concentration range: {true_concentrations.min():.4f} to {true_concentrations.max():.4f} mg/L")
    print(f"- Dosing times (days): {list(dosing_schedule['Day'].values)}")
    print(f"- Dose amounts: {list(dosing_schedule['Dose_Amount (mg)'].values)}")
    
    # Plot the data
    plt.figure(figsize=(12, 6))
    plt.plot(pk_data['Day'], pk_data['True_Concentration (mg/L)'], '-', 
             label='True Concentration', linewidth=2, color='blue')
    plt.plot(pk_data['Day'], pk_data['Concentration_with_noise (mg/L)'], 'o', 
             alpha=0.6, markersize=3, label='Observed (with noise)', color='orange')
    
    # Add vertical lines for dosing events
    for i, (day, dose) in dosing_schedule.iterrows():
        plt.axvline(x=day, color='red', linestyle='--', alpha=0.7)
        plt.annotate(f'{dose:.0f} mg', xy=(day, pk_data['True_Concentration (mg/L)'].max()*0.8), 
                    xytext=(day+0.05, pk_data['True_Concentration (mg/L)'].max()*0.9),
                    arrowprops=dict(arrowstyle='->', color='red'))
    
    plt.xlabel('Time (days)')
    plt.ylabel('Concentration (mg/L)')
    plt.title('PK Data from 2C_simulator.py (Converted for PyTorch)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    images_dir = "../../images"
    os.makedirs(images_dir, exist_ok=True)
    plt.savefig(os.path.join(images_dir, 'pytorch_compatible_pk_data.png'), 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nFiles created:")
    print(f"- data/pk_simulation_data.csv")
    print(f"- data/dosing_schedule.csv") 
    print(f"- images/pytorch_compatible_pk_data.png")
    
    return pk_data, dosing_schedule

if __name__ == "__main__":
    convert_npz_to_pytorch_csv()