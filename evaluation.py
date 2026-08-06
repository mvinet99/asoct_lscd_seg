# Perform downstream evaluation on the output predictions from the model

# Import modules
import numpy as np
import matplotlib.pyplot as plt
import statistics
import os
import matplotlib
import random
import pandas as pd
import scipy
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)

def get_triple_sums_list(a_list):
    """
    Given a list, return a new list where each element is the 
    average of every three consecutive elements from the original list.
    """

    new_list = []
    for index in range(0,len(a_list), 3):
        total = sum(a_list[index:index+3])/3
                                         
        new_list.append(total)
    return new_list

def column_to_list(df, column_name):
    """
    Take DataFrame and convert a specified column into a list.
    """

    if column_name in df.columns:
        return df[column_name].tolist()
    else:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

def evaluate(result_path, datasheet_path):
    """
    Evaluate the predictions made by the model and generate visualizations.
    """

    df1 = pd.read_excel(datasheet_path) 
    file_names = df1["File Name"].tolist()
    epi_thicknesses = df1["Epithelial Thickness"].tolist()
    optovue_thicknesses = df1["OCT-CET Thickness"].tolist()
    severities = df1["Severity"].tolist()

    # Load the slice name and average thickness number - make it so that it first checks for an actual value
    file_path = result_path + 'thickness_npy/'
    files = os.listdir(file_path)

    control_vals = []
    mild_vals = []
    moderate_vals = []
    severe_vals = []
    c_control_vals = []
    c_mild_vals = []
    c_moderate_vals = []
    c_severe_vals = []
    o_control_vals = []
    o_mild_vals = []
    o_moderate_vals = []
    o_severe_vals = []
    cr_control_vals = []
    cr_mild_vals = []
    cr_moderate_vals = []
    cr_severe_vals = []
    cro_control_vals = []
    cro_mild_vals = []
    cro_moderate_vals = []
    cro_severe_vals = []

    for file in tqdm(files):
        file_temp = file[0:-8]
        severity = severities[file_names.index(file_temp)]
        epi_thickness = epi_thicknesses[file_names.index(file_temp)]
        opt_thickness = optovue_thicknesses[file_names.index(file_temp)]

        if severity == 'control' or severity == 'Control':
            if epi_thickness == ' ':
                control_vals.append(statistics.mean(np.load(file_path+file)))
                o_control_vals.append(opt_thickness)
            else:
                control_vals.append(statistics.mean(np.load(file_path+file)))
                c_control_vals.append(epi_thickness)
                o_control_vals.append(opt_thickness)
                cr_control_vals.append(statistics.mean(np.load(file_path+file)))
                cro_control_vals.append(opt_thickness)

        elif severity == 'mild' or severity == 'Mild':
            if epi_thickness == ' ':
                mild_vals.append(statistics.mean(np.load(file_path+file)))
                o_mild_vals.append(opt_thickness)
            else:
                mild_vals.append(statistics.mean(np.load(file_path+file)))
                c_mild_vals.append(epi_thickness)
                o_mild_vals.append(opt_thickness)
                cr_mild_vals.append(statistics.mean(np.load(file_path+file)))
                cro_mild_vals.append(opt_thickness)
        elif severity == 'moderate' or severity == 'Moderate':
            if epi_thickness == ' ':
                moderate_vals.append(statistics.mean(np.load(file_path+file)))
                o_moderate_vals.append(opt_thickness)
            else:
                moderate_vals.append(statistics.mean(np.load(file_path+file)))
                c_moderate_vals.append(epi_thickness)
                o_moderate_vals.append(opt_thickness)
                cr_moderate_vals.append(statistics.mean(np.load(file_path+file)))
                cro_moderate_vals.append(opt_thickness)
        elif severity == 'severe' or severity == 'Severe':
            if epi_thickness == ' ':
                severe_vals.append(statistics.mean(np.load(file_path+file)))
                o_severe_vals.append(opt_thickness)
            else:
                severe_vals.append(statistics.mean(np.load(file_path+file)))
                c_severe_vals.append(epi_thickness)
                o_severe_vals.append(opt_thickness)
                cr_severe_vals.append(statistics.mean(np.load(file_path+file)))
                cro_severe_vals.append(opt_thickness)

    # BOX PLOT - All Classes
    data_sets = [
        c_control_vals, c_mild_vals, c_moderate_vals, c_severe_vals, # M-CET
        control_vals, mild_vals, moderate_vals, severe_vals, # AI-CE
        o_control_vals, o_mild_vals, o_moderate_vals, o_severe_vals] # OCT-CET

    labels = [
        'M-CET Control', 'M-CET Mild', 'M-CET Moderate', 'M-CET Severe',
        'AI-CE Control', 'AI-CE Mild', 'AI-CE Moderate', 'AI-CE Severe',
        'OCT-CET Control', 'OCT-CET Mild', 'OCT-CET Moderate', 'OCT-CET Severe'
    ]

    # Define colors for the boxes by condition
    colors = ['green','blue','orange','red', 'green','blue','orange','red','green','blue','orange','red']

    # Professional white grid theme
    sns.set_theme(style="whitegrid")  

    # Custom colors for categories
    # Using seaborn color palette
    palette = sns.color_palette("husl", 5)  
    category_colors = {
        "Control": palette[2],
        "Mild": palette[3],
        "Moderate": palette[1],
        "Severe": palette[0]}

    # Create the figure
    plt.figure(figsize=(12, 7))

    # Create the boxplot
    positions = range(1, len(data_sets) + 1)
    positions = [0.2, 0.6, 1.0, 1.4, 2.2, 2.6, 3.0, 3.4, 4.2, 4.6, 5.0, 5.4] 
    boxes = plt.boxplot(
        data_sets, 
        patch_artist=True, 
        notch=False, 
        vert=True,
        widths=0.1, 
        positions=positions, 
        boxprops=dict(edgecolor="dimgray", linewidth=1.5),
        medianprops=dict(color="black", linewidth=2),
        capprops=dict(color="dimgray", linewidth=1.5),
        whiskerprops=dict(color="dimgray", linewidth=1.5),
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=6, linestyle="none")
    )

    # Apply colors and thicker medians
    for i, box in enumerate(boxes['boxes']):
        r, g, b = matplotlib.colors.to_rgb(colors[i])
        box.set_facecolor((r, g, b, 0.6))
        box.set_edgecolor('black')  # Define edges clearly
        box.set_linewidth(1.2)

    # Customize x-ticks
    plt.xticks(
        ticks=positions, 
        labels=labels, 
        rotation=45, 
        ha="right", 
        fontsize=12
    )

    # Customize axis labels and title
    plt.ylabel("CET (\u03bcm)", fontsize=14, fontweight="bold")
    plt.title("CET by Severity", fontsize=16, fontweight="bold")

    # Improve grid aesthetics
    plt.grid(True, linestyle="--", alpha=0.7)

    # Set y-axis limits
    plt.ylim([-5, 77])

    # Improve layout and save
    plt.tight_layout()
    plt.savefig(result_path + "eval_images/BoxPlot.png", dpi=300)
    plt.close('all')

    clinic_all = c_control_vals + c_mild_vals + c_moderate_vals + c_severe_vals
    all_avgs = control_vals + mild_vals + moderate_vals + severe_vals
    opto_all = cro_control_vals + cro_mild_vals + cro_moderate_vals + cro_severe_vals

    # Correlation plot
    plt.figure(figsize=(8, 8))
    labels1 = ['Control', 'Mild', 'Moderate', 'Severe']
    i = 0
    for category, color in category_colors.items():
        x_values = eval(f"c_{category.lower()}_vals")
        y_values = eval(f"{category.lower()}_vals")
        plt.scatter(x_values, y_values, color=color, alpha=0.8, label=labels1[i], edgecolors='black',s=80)
        i = i + 1

    # Regression line
    slope, intercept, r, _, _ = scipy.stats.linregress(clinic_all, all_avgs)
    x = np.linspace(min(clinic_all), max(clinic_all), 100)
    y = slope * x + intercept

    # Plot regression line
    plt.plot(x, y, color="black", linestyle="--", label=f"Best Fit (r = "+str(r)[0:5]+")")
    plt.rcParams['font.size'] = 20
    plt.xlabel("M-CET (\u03bcm)", fontsize=18, fontweight='bold')
    plt.ylabel("AI-CET (\u03bcm)", fontsize=18, fontweight='bold')
    plt.legend(fontsize=16, loc='upper left')
    plt.xlim([-1, 72])
    plt.ylim([-1, 81])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(result_path + "eval_images/Correlation_AI_CE.png", dpi=300)
    plt.close('all')

    # Correlation plot - OCT-CET vs M-CET
    plt.figure(figsize=(8, 8))
    for category, color in category_colors.items():
        x_values = eval(f"c_{category.lower()}_vals")
        y_values = eval(f"cro_{category.lower()}_vals")
        plt.scatter(x_values, y_values, color=color, alpha=0.8, edgecolors='black', s=80)

    # OCT-CET Correlation
    slope, intercept, r2_tk, _, _ = scipy.stats.linregress(clinic_all, opto_all)
    x = np.linspace(min(clinic_all), max(clinic_all), 100)
    y = slope * x + intercept
    plt.plot(x, y, color="black", linestyle="--", label=f"Best Fit (r = "+str(r2_tk)[0:5]+")")
    plt.rcParams['font.size'] = 20
    plt.xlabel("M-CET (\u03bcm)", fontsize=18, fontweight='bold')
    plt.ylabel("OCT-CET (\u03bcm)", fontsize=18, fontweight='bold')
    plt.legend(fontsize=16, loc='upper left')

    plt.xlim([-1, 72])
    plt.ylim([-1, 81])
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.savefig(result_path + "eval_images/Correlation_OCT.png", dpi=300)
    plt.close('all')

    # Calculate MAE
    mae = mean_absolute_error(clinic_all, all_avgs)
    _, _, r_all, _, _ = scipy.stats.linregress(clinic_all, all_avgs)
    mae_control = mean_absolute_error(c_control_vals, control_vals)
    _, _, r_control, _, _ = scipy.stats.linregress(c_control_vals, control_vals)
    mae_mild = mean_absolute_error(c_mild_vals, mild_vals)
    _, _, r_mild, _, _ = scipy.stats.linregress(c_mild_vals, mild_vals)
    mae_moderate = mean_absolute_error(c_moderate_vals, moderate_vals)
    _, _, r_moderate, _, _ = scipy.stats.linregress(c_moderate_vals, moderate_vals)
    mae_severe = mean_absolute_error(c_severe_vals, severe_vals)
    _, _, r_severe, _, _ = scipy.stats.linregress(c_severe_vals, severe_vals)

    # Print result
    print(f"Average patch MAE: {mae:.4f}", f"Pearson r: {r_all:.4f}")
    print(f"Control patch MAE: {mae_control:.4f}", f"Pearson r: {r_control:.4f}")
    print(f"Mild patch MAE: {mae_mild:.4f}", f"Pearson r: {r_mild:.4f}")
    print(f"Moderate patch MAE: {mae_moderate:.4f}", f"Pearson r: {r_moderate:.4f}")
    print(f"Severe patch MAE: {mae_severe:.4f}", f"Pearson r: {r_severe:.4f}")
    