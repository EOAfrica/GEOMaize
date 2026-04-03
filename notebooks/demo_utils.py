from matplotlib.patches import Patch
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import numpy as np
from openeo_gfmap import BoundingBoxExtent
from openeo_gfmap import TemporalContext
from datetime import datetime
import xarray as xr

def plot_distribution(train_df, test_df, val_df, target_column):
    # Create a figure with subplots for the value counts including train/test/val splits
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Distribution of Variables by Train/Test/Validation Split', fontsize=16)

    # Define colors for train/test/val
    colors = ['blue', 'orange', 'green']
    labels = ['Train', 'Test', 'Val']
    datasets = [train_df, test_df, val_df]

    # REGION distribution
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        region_counts = data.REGION.value_counts()
        axes[0, 0].bar(region_counts.index, region_counts.values, alpha=0.7, color=color, label=label)
    axes[0, 0].set_title('REGION')
    axes[0, 0].set_xlabel('Region')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].legend()

    # COMMUNITY distribution
    all_communities = pd.concat(datasets).COMMUNITY.unique()
    x_pos = range(len(all_communities))
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        community_counts = data.COMMUNITY.value_counts().reindex(all_communities, fill_value=0)
        axes[0, 1].bar([x + i*0.25 for x in x_pos], community_counts.values, width=0.25, alpha=0.7, color=color, label=label)
    axes[0, 1].set_title('COMMUNITY')
    axes[0, 1].set_xlabel('Community')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_xticks([x + 0.25 for x in x_pos])
    axes[0, 1].set_xticklabels(all_communities, rotation=45, ha='right')
    axes[0, 1].legend()

    # DISTRICT distribution
    all_districts = pd.concat(datasets).DISTRICT.unique()
    x_pos_district = range(len(all_districts))
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        district_counts = data.DISTRICT.value_counts().reindex(all_districts, fill_value=0)
        axes[1, 0].bar([x + i*0.25 for x in x_pos_district], district_counts.values, width=0.25, alpha=0.7, color=color, label=label)
    axes[1, 0].set_title('DISTRICT')
    axes[1, 0].set_xlabel('District')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_xticks([x + 0.25 for x in x_pos_district])
    axes[1, 0].set_xticklabels(all_districts, rotation=45, ha='right')
    axes[1, 0].legend()

    # YEAR distribution
    all_years = sorted(pd.concat(datasets).year.unique())
    x_pos_year = range(len(all_years))
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        year_counts = data.year.value_counts().reindex(all_years, fill_value=0)
        axes[1, 1].bar([x + i*0.25 for x in x_pos_year], year_counts.values, width=0.25, alpha=0.7, color=color, label=label)
    axes[1, 1].set_title('YEAR')
    axes[1, 1].set_xlabel('Year')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_xticks([x + 0.25 for x in x_pos_year])
    axes[1, 1].set_xticklabels(all_years)
    axes[1, 1].legend()

    # Yield kg/H histogram with bars next to each other
    bin_edges = np.histogram_bin_edges(
        pd.concat([train_df[target_column], test_df[target_column], val_df[target_column]]).dropna(), bins=20
    )

    width = 0.25
    # Plot histogram for Yield kg/H for each split using the same bin edges
    for i, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        hist, _ = np.histogram(data[target_column].dropna(), bins=bin_edges)
        # Use bin centers for bar positions
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        axes[0, 2].bar(bin_centers + i*width - width, hist, width=width, alpha=0.7, color=color, label=label, align='center')
        # x_ticks = [bin_centers + i*width]
    axes[0, 2].set_title('Bin Yield kg/H')
    # axes[0, 2].set_xlabel('Yield kg/H')
    # axes[0, 2].set_xticks(x_ticks)
    axes[0, 2].set_xticklabels(['', '', 'Low', '', '', '', 'High', ''])
    axes[0, 2].set_ylabel('Frequency')
    axes[0, 2].legend()

    # Bin mean yield and std plot
    bin_labels = ['Low', 'High'] if 'bin' in train_df.columns else train_df['bin'].unique().astype(str)
    x = np.arange(len(bin_labels))
    for idx, (data, color, label) in enumerate(zip(datasets, colors, labels)):
        bin_means = data.groupby('bin')['Yield kg/H'].mean().reindex([0,1], fill_value=0)
        bin_stds = data.groupby('bin')['Yield kg/H'].std().reindex([0,1], fill_value=0)
        axes[1, 2].bar(x + idx*width, bin_means.values, width=width, yerr=bin_stds.values, capsize=5, color=color, alpha=0.8, label=label)

    axes[1, 2].set_title('Bin Mean Yield ± Std')
    axes[1, 2].set_xlabel('Bin')
    axes[1, 2].set_ylabel('Mean Yield (kg/H)')
    axes[1, 2].set_xticks(x + width)
    axes[1, 2].set_xticklabels(bin_labels)
    axes[1, 2].legend()

    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(train_preds, val_preds, test_preds, train_targets, val_targets, test_targets):
    model_name = "Presto"
    split_names = ["Train", "Validation", "Test"]
    model_preds = [(preds > 0.5).astype(int) if preds.dtype == np.float32 else preds for preds in [train_preds, val_preds, test_preds]]
    model_trues = [train_targets, val_targets, test_targets]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for col, split_name in enumerate(split_names):
        y_true = model_trues[col]
        y_pred = model_preds[col]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[col], cbar=False,
                    xticklabels=["Low", "High"], yticklabels=["Low", "High"])
        axes[col].set_xlabel('Predicted')
        axes[col].set_ylabel('True')
        axes[col].set_title(f"{model_name} - {split_name}")

    plt.tight_layout()
    plt.show()
    
def plot_yield_prediction_vs_target(
    train_preds, 
    val_preds,
    test_preds, 
    train_targets, 
    val_targets, 
    test_targets, 
    train_df,
    val_df,
    test_df,
    target_name="Yield kg/H", 
    bin_th=1220):
    plot_data = [
    (
        "Presto",
        [train_targets, val_targets, test_targets],
        [train_preds, val_preds, test_preds],
        [train_df[target_name].values, val_df[target_name].values, test_df[target_name].values],
    ),
    ]

    col_titles = ["Train", "Validation", "Test"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=False, sharey=True)

    for row, (model_name, y_trues, y_preds, yields) in enumerate(plot_data):
        for col in range(3):
            ax = axes[col]
            y_true = y_trues[col]
            y_pred = y_preds[col]
            yield_vals = yields[col]
            # y_pred_int = (y_pred > 0.5).astype(int) if np.issubdtype(y_pred.dtype, np.floating) else y_pred.astype(int)
            scatter = ax.scatter(
                np.arange(len(yield_vals)),
                yield_vals,
                c=y_pred,
                cmap="coolwarm",
                vmin=0,
                vmax=1,
                alpha=0.7,
                marker="o",
                )
            # Draw the bin threshold line and always show its label in the legend
            ax.axhline(bin_th, color="black", linestyle="--", label="Low/High threshold")
            ax.set_title(f"{model_name} - {col_titles[col]}")
            ax.set_xlabel("Sample Index")
        if col == 0:
            ax.set_ylabel(target_name)
        if row == 0 and col == 2:
            legend_elements = [
                Patch(facecolor=plt.cm.coolwarm(0.0), label="Low Yield"),
                Patch(facecolor=plt.cm.coolwarm(1.0), label="High Yield"),
                plt.Line2D([0], [0], color="black", linestyle="--", label="Low/High threshold"),
            ]
        ax.legend(handles=legend_elements, loc="upper right")

def get_spatial_and_temporal_extents(extent, start_date, end_date, epsg=4326):
    minx, miny, maxx, maxy = extent.total_bounds
    spatial_context = BoundingBoxExtent(
        west = float(minx),
        south = float(miny),
        east = float(maxx),
        north = float(maxy),
        epsg=epsg
    )
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")
    temporal_context = TemporalContext(start_date=start_date, end_date=end_date)
    return spatial_context, temporal_context

def min_max_normalize(image, gamma=1.0):
    # normalize image and set NaNs to NODATA value
    image = np.nan_to_num(image, 65535).astype("uint16")
    image = (image - image.min()) / (image.max() - image.min()) * gamma
    image = np.clip(image, 0, 1)  # Ensure values are between 0 and 1 after applying gamma
    return image

def plot_results(
        path_to_input_file, 
        prob_map=None, 
        bin_th=0.5,
        ts_index=0,
        coords = None, 
        rgb_gamma=1.0,
        prob_cmap="Greens", 
        ):

    ts_index=3
    rgb_gamma=1 
    rgb = xr.load_dataset(path_to_input_file)
    if coords is not None:
        rgb = rgb.isel(x=slice(coords[0], coords[2]), y=slice(coords[1], coords[3]))
    bands = ["S2-L2A-B04", "S2-L2A-B03", "S2-L2A-B02"]
    rgb = np.stack([rgb[band].values for band in bands], axis=-1)[ts_index]

    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05], wspace=0.1)

    axs = [fig.add_subplot(gs[i]) for i in range(3)]
    cax = fig.add_subplot(gs[3])  # dedicated colorbar axis

    axs[0].imshow(min_max_normalize(rgb, gamma=rgb_gamma))
    axs[0].set_title("RGB")
    axs[0].axis("off")

    rgb_prediction_map = np.zeros((*prob_map.shape, 3), dtype=np.float32)
    rgb_prediction_map[rgb_prediction_map == 0] = np.nan

    valid_mask = prob_map != 0
    high_mask = valid_mask & (prob_map > bin_th)
    low_mask = valid_mask & (prob_map <= bin_th)

    rgb_prediction_map[high_mask] = [0, 1, 0]  # green
    rgb_prediction_map[low_mask] = [1, 0, 0]   # red

    # Replace only fully black pixels ([0, 0, 0]) with RGB background values
    background_mask = np.all(np.isnan(rgb_prediction_map), axis=-1)
    rgb_bg = rgb.astype(np.float32)
    if np.nanmax(rgb_bg) > 1:
        rgb_bg = rgb_bg / np.nanmax(rgb_bg)

    rgb_prediction_map[background_mask] = rgb_bg[background_mask]
    axs[1].imshow(rgb_prediction_map)
    axs[1].set_title(f"Prediction Map > {bin_th}")
    axs[1].axis("off")


    im = axs[2].imshow(prob_map, cmap=prob_cmap, vmin=0, vmax=1)
    axs[2].set_title("Probability Map")
    axs[2].axis("off")

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_ticks(np.arange(0, 1.1, 0.1))
    plt.show()