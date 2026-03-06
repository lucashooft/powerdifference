from fitparse import FitFile
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math

# top-level data/merged directories
DATA_DIR = Path('./data')
MERGED_DIR = Path('./mergeddata')
PLOT_DIR = Path('./plot')
TRANSLATED_DIR = Path('./translateddata')
TRANSPLOT_DIR = Path('./translatedplot')

MERGED_DIR.mkdir(parents=True, exist_ok=True)
PLOT_DIR.mkdir(parents=True, exist_ok=True)
TRANSLATED_DIR.mkdir(parents=True, exist_ok=True)
TRANSPLOT_DIR.mkdir(parents=True, exist_ok=True)

if not DATA_DIR.exists():
    print({'error': f'data directory not found: {DATA_DIR}'})
else:
    # Process each subject folder: expect exactly 2 .fit files per subject
    for sub in sorted(p for p in DATA_DIR.iterdir() if p.is_dir()):
        fits = sorted(sub.glob('*.fit'))
        if len(fits) != 2:
            print({'subdir': str(sub), 'error': f'expected 2 fit files, found {len(fits)}'})

            # special-case: only 4iiii.fit available -> create dataframe and try to adjust using merged.csv
            if len(fits) == 1 and fits[0].name == '4iiii.fit':
                d_single = {}
                try:
                    fit = FitFile(str(fits[0]))
                    for message in fit.get_messages('record'):
                        ts = message.get_value('timestamp')
                        p = message.get_value('power')
                        if ts is None:
                            continue
                        d_single[ts] = p
                except Exception as e:
                    print({'subdir': str(sub), 'file': str(fits[0]), 'error': str(e)})
                    # continue to next subject
                    continue

                if not d_single:
                    print({'subdir': str(sub), 'info': 'no records in 4iiii.fit; skipping'})
                    continue

                # build dataframe with timestamp and 4iiii power
                times = sorted(d_single.keys())
                df_single = pd.DataFrame({
                    'timestamp': times,
                    '4iiii': [d_single[ts] for ts in times]
                })

                # try to read merged reference (mean per kickr) to obtain %diff mapping
                merged_ref = MERGED_DIR / 'merged.csv'
                if not merged_ref.exists():
                    # just save the raw 4iiii dataframe
                    out_csv_single = MERGED_DIR / f'{sub.name}_4iiii_only.csv'
                    df_single.to_csv(out_csv_single, index=False)
                    print({'subdir': str(sub), 'info': f'Wrote raw 4iiii CSV (no merged reference): {out_csv_single}'})
                    continue

                try:
                    ref = pd.read_csv(merged_ref)
                except Exception as e:
                    print({'subdir': str(sub), 'error': f'failed to read {merged_ref}: {e}'})
                    out_csv_single = MERGED_DIR / f'{sub.name}_4iiii_only.csv'
                    df_single.to_csv(out_csv_single, index=False)
                    print({'subdir': str(sub), 'info': f'Wrote raw 4iiii CSV due to read error: {out_csv_single}'})
                    continue

                # expect ref to contain 'kickr' and '%diff' columns
                if 'kickr' not in ref.columns or '%diff' not in ref.columns:
                    out_csv_single = MERGED_DIR / f'{sub.name}_4iiii_only.csv'
                    df_single.to_csv(out_csv_single, index=False)
                    print({'subdir': str(sub),
                           'info': f'{merged_ref} missing required columns; wrote raw 4iiii CSV: {out_csv_single}'})
                    continue

                ref['kickr'] = pd.to_numeric(ref['kickr'], errors='coerce')
                ref['%diff'] = pd.to_numeric(ref['%diff'], errors='coerce')
                ref = ref.dropna(subset=['kickr', '%diff']).reset_index(drop=True)
                if ref.empty:
                    out_csv_single = MERGED_DIR / f'{sub.name}_4iiii_only.csv'
                    df_single.to_csv(out_csv_single, index=False)
                    print({'subdir': str(sub),
                           'info': f'no valid rows in {merged_ref}; wrote raw 4iiii CSV: {out_csv_single}'})
                    continue


                # helper: find nearest %diff in ref for a given 4iiii value by matching nearest kickr
                def nearest_pct(val):
                    try:
                        idx = (ref['kickr'] - val).abs().idxmin()
                        return float(ref.loc[idx, '%diff'])
                    except Exception:
                        return np.nan


                # compute nearest %diff and estimated kickr = 4iiii / (1 + %diff/100)
                df_single['%diff_ref'] = df_single['4iiii'].apply(nearest_pct)
                df_single['kickr_est'] = df_single.apply(
                    lambda r: (r['4iiii'] / (1.0 + r['%diff_ref'] / 100.0)) if pd.notna(r['%diff_ref']) else np.nan,
                    axis=1
                )

                # round/cast estimated kickr
                df_single['kickr_est'] = pd.to_numeric(df_single['kickr_est'], errors='coerce').round(0).astype('Int64')

                # keep rows with an estimate and preserve timestamp for plotting
                df_plot = df_single.dropna(subset=['kickr_est']).copy()
                if df_plot.empty:
                    print({'subdir': str(sub), 'info': 'no estimated kickr values; skipping output'})
                    continue

                # remove duplicate 4iiii values, keep the first, and sort by timestamp
                df_plot = df_plot.drop_duplicates(subset=['4iiii']).sort_values('timestamp').reset_index(drop=True)

                # for CSV output the user asked to keep only 4iiii and kickr_est
                df_out = df_plot[['4iiii', 'kickr_est']].copy()
                out_est = TRANSLATED_DIR / f'{sub.name}_4iiii_est.csv'
                df_out.to_csv(out_est, index=False)
                print({'subdir': str(sub), 'info': 'wrote 4iiii adjusted estimates (4iiii + kickr_est)',
                       'file': str(out_est)})

                # create a timeseries plot with both values on the y-axis
                try:
                    fig, ax = plt.subplots(figsize=(8, 4))
                    # x = timestamps for plotting
                    x = pd.to_datetime(df_plot['timestamp']) if 'timestamp' in df_plot.columns else df_plot.index
                    ax.plot(x, df_plot['4iiii'].astype(float), marker='o', linestyle='-', label='4iiii')
                    ax.plot(x, df_plot['kickr_est'].astype(float), marker='o', linestyle='--', label='kickr_est')
                    ax.set_title(f'4iiii and estimated kickr — {sub.name}')
                    ax.set_xlabel('timestamp')
                    ax.set_ylabel('power')
                    ax.grid(True)
                    ax.legend()
                    fig.autofmt_xdate()
                    plot_path = TRANSPLOT_DIR / f'{sub.name}_4iiii_kickr_est.png'
                    fig.tight_layout()
                    fig.savefig(plot_path)
                    plt.close(fig)
                    print({'subdir': str(sub), 'info': 'wrote 4iiii vs kickr_est plot', 'plot': str(plot_path)})
                except Exception as e:
                    print({'subdir': str(sub), 'error': f'failed to create plot: {e}'})

            # end special-case for 4iiii only
            continue

        powers = []
        for path in fits:
            try:
                fit = FitFile(str(path))
            except Exception as e:
                print({'subdir': str(sub), 'file': str(path), 'error': str(e)})
                fit = None

            d = {}
            if fit is not None:
                for message in fit.get_messages('record'):
                    ts = message.get_value('timestamp')
                    p = message.get_value('power')
                    if ts is None:
                        continue
                    d[ts] = p
            powers.append(d)

        # merge timestamps
        try:
            all_timestamps = sorted(set().union(*powers))
        except Exception as e:
            print({'subdir': str(sub), 'error': f'failed to union timestamps: {e}'})
            continue

        df = pd.DataFrame()
        df['4iiii'] = [powers[0].get(ts) for ts in all_timestamps]
        df['kickr'] = [powers[1].get(ts) for ts in all_timestamps]

        df = df.dropna(how='any')
        df = df.reset_index().drop(columns=['index'])
        df.index.name = 'index'

        # remove zero kickr rows to avoid division by zero
        df = df[df['kickr'] != 0]

        if df.empty:
            print({'subdir': str(sub), 'info': 'no valid rows after dropping NaN/zero kickr; skipping'})
            continue

        df['%diff'] = ((df['4iiii'] - df['kickr']) / df['kickr'] * 100).round(0).astype('Int64')

        # write base merged CSV for this subject
        out_csv = MERGED_DIR / f'{sub.name}.csv'
        df.to_csv(out_csv, index=False)

        # per-10 bin aggregation (keep bins with at least 100 samples)
        df2 = df.copy()
        df2['kickr10'] = ((df2['kickr'] // 10) * 10).astype('Int64')
        agg = df2.groupby('kickr10')['%diff'].agg(['mean', 'count']).reset_index()
        agg = agg[agg['count'] >= 50].copy()

        if agg.empty:
            print({'subdir': str(sub), 'info': 'no kickr10 bins with >=100 samples; skipping per-10 CSV and plot'})
        else:
            agg = agg.rename(columns={'mean': '%diff'})
            per10 = agg[['kickr10', '%diff', 'count']]
            # sort and remove duplicate kickr10 values if any
            per10 = per10.drop_duplicates(subset=['kickr10']).sort_values('kickr10').reset_index(drop=True)
            per10_csv = MERGED_DIR / f'{sub.name}_per10.csv'
            per10.to_csv(per10_csv, index=False)

            # plot per-subject per-10 with trend line and annotations
            fig, ax = plt.subplots(figsize=(6, 4))
            x = per10['kickr10'].astype(float)
            y = per10['%diff'].astype(float)
            ax.plot(x, y, marker='o', linestyle='-')
            # trend line when at least 2 points
            if len(x) >= 2:
                coeffs = np.polyfit(x, y, 2)
                x_sorted = np.sort(x)
                y_trend = np.polyval(coeffs, x_sorted)
                ax.plot(x_sorted, y_trend, color='C1', linestyle='--')
            # annotate each plotted point (rounded) — show kickr * (1 + %diff/100)
            for xi, yi in zip(x, y):
                try:
                    val = xi * (1 + yi / 100.0)
                    ax.annotate(f'{val:.0f}', (xi, yi), textcoords='offset points', xytext=(0, 6), ha='center',
                                fontsize=8)
                except Exception:
                    pass

            ax.set_title(f'Percent diff vs kickr10 — {sub.name}')
            ax.set_xlabel('kickr10')
            ax.set_ylabel('%diff')
            ax.grid(True)
            fig.tight_layout()
            plot_path = PLOT_DIR / f'{sub.name}_per10_plt.png'
            fig.savefig(plot_path)
            plt.close(fig)
            print({'subdir': str(sub), 'info': f'wrote per-10 CSV and plot for subject', 'per10_csv': str(per10_csv),
                   'plot': str(plot_path)})

# Combine all base CSVs (those not ending with _per10.csv) into one merged dataset and produce combined per-10
# Only run combine step if merged_dir exists (it does) and contains CSVs
csv_files = sorted(p for p in MERGED_DIR.glob('*.csv') if not p.name.endswith('_per10.csv'))
if not csv_files:
    print({'info': 'no base CSV files in ./mergeddata to combine'})
else:
    combined_dfs = []
    for p in csv_files:
        try:
            df_tmp = pd.read_csv(p)
        except Exception as e:
            print({'file': str(p), 'error': f'failed to read: {e}'})
            continue
        # require columns
        if 'kickr' not in df_tmp.columns or '%diff' not in df_tmp.columns:
            print({'file': str(p), 'info': 'skipping: missing required columns (kickr, %diff)'})
            continue
        combined_dfs.append(df_tmp[['kickr', '%diff']].copy())

    if not combined_dfs:
        print({'info': 'no valid CSVs to combine after filtering columns'})
    else:
        combined = pd.concat(combined_dfs, ignore_index=True)
        combined['kickr'] = pd.to_numeric(combined['kickr'], errors='coerce')
        combined['%diff'] = pd.to_numeric(combined['%diff'], errors='coerce')
        combined = combined.dropna(subset=['kickr', '%diff'])
        combined = combined[combined['kickr'] != 0]

        combined_csv = MERGED_DIR / 'merged.csv'
        # produce mean (%diff) per exact kickr value (with count) for merged.csv
        grouped = combined.groupby('kickr')['%diff'].agg(['mean', 'count']).reset_index()
        grouped = grouped[grouped['mean'] <= 40].copy()
        grouped = grouped[grouped['count'] >= 50].copy()
        grouped = grouped.rename(columns={'mean': '%diff'})
        # sort and drop duplicate kickr values (keep first)
        grouped = grouped.drop_duplicates(subset=['kickr']).sort_values('kickr').reset_index(drop=True)
        grouped.to_csv(combined_csv, index=False)
        print({'info': f'Wrote combined CSV (mean per kickr): {combined_csv}'})

        # plot combined exact-kickr grouped mean with trend and annotations
        fig, ax = plt.subplots(figsize=(8, 5))
        xg = grouped['kickr'].astype(float)
        yg = grouped['%diff'].astype(float)
        ax.plot(xg, yg, marker='o', linestyle='-')
        if len(xg) >= 2:
            coeffs = np.polyfit(xg, yg, 2)
            xg_sorted = np.sort(xg)
            yg_trend = np.polyval(coeffs, xg_sorted)
            ax.plot(xg_sorted, yg_trend, color='C1', linestyle='--')
        #  for xi, yi in zip(xg, yg):
        #     try:
        #        val = xi * (1 + yi / 100.0)
        #       ax.annotate(f'{val:.0f}', (xi, yi), textcoords='offset points', xytext=(0, 6), ha='center', fontsize=8)
        #  except Exception:
        #     pass

        ax.set_title('Combined percent diff vs kickr')
        ax.set_xlabel('kickr')
        ax.set_ylabel('%diff')
        ax.grid(True)
        fig.tight_layout()
        plot_path = PLOT_DIR / 'total_plt.png'
        fig.savefig(plot_path)
        plt.close(fig)
        print({'info': f'Wrote combined plot: {plot_path}'})

ftp = 297

# Kickr-values are percentages of FTP ranges (zones defined as % of FTP)
zones = {
    "Active Recovery (Z1)": (0, 55),
    "Endurance (Z2)": (55, 75),
    "Tempo (Z3)": (75, 87),
    "Sweet Spot (Z4)": (87, 94),
    "Threshold (Z5)": (94, 105),
    "VO2 Max (Z6)": (105, 120),
    "Anaerobic Capacity (Z7)": (120, 10_000)
}


# assign zone based on percentage-of-FTP value
def zone_for_value(v):
    for name, (lo, hi) in zones.items():
        if lo <= v < hi:
            return name
    return None


# Use the grouped dataframe (created above) which contains 'kickr', '%diff' and 'count'
# If grouped isn't available (e.g. earlier steps skipped), skip zone plotting gracefully.
grouped_var = globals().get('grouped', locals().get('grouped', None))
if grouped_var is None:
    print({'info': 'grouped dataframe not available; skipping zone stats/plot'})
else:
    # work on a copy to avoid mutating original
    if hasattr(grouped_var, 'copy'):
        g = grouped_var.copy()
    else:
        g = pd.DataFrame(grouped_var)
    # ensure numeric
    g['kickr'] = pd.to_numeric(g['kickr'], errors='coerce')
    g['%diff'] = pd.to_numeric(g['%diff'], errors='coerce')
    g['count'] = pd.to_numeric(g['count'], errors='coerce').fillna(0).astype(int)
    g = g.dropna(subset=['kickr', '%diff'])
    if g.empty:
        print({'info': 'grouped dataframe empty after cleanup; skipping zone stats/plot'})
    else:
        # convert kickr to percent of FTP
        g['pct_ftp'] = g['kickr'].astype(float) / float(ftp) * 100.0
        g['zone'] = g['pct_ftp'].apply(zone_for_value)
        g = g.dropna(subset=['zone'])
        if g.empty:
            print({'info': 'no rows assigned to a power zone; skipping zone stats/plot'})
        else:
            # weighted average of %diff per zone using 'count' as weights
            def weighted_avg(series):
                vals = series['%diff']
                weights = series['count']
                if weights.sum() == 0:
                    return np.nan
                return (vals * weights).sum() / weights.sum()


            # compute weighted average of %diff per zone; explicitly select columns to avoid
            # FutureWarning about apply operating on grouping columns
            zone_stats = (
                g.groupby('zone')[['%diff', 'count']]
                .apply(lambda df: weighted_avg(df))
                .to_frame('weighted_avg_diff')
                .reset_index()
                .sort_values('zone')
            )

            # compute zone midpoint (zonevalue) and requested y-values
            # zone_mid_pct is midpoint of zone in percent of FTP (e.g. 65 for 55-75)
            zone_mid_pct_map = {name: (lo + hi) / 2.0 for name, (lo, hi) in zones.items()}
            zone_stats['zone_mid_pct'] = zone_stats['zone'].map(zone_mid_pct_map)
            zone_stats['zone_mid_frac'] = zone_stats['zone_mid_pct'] / 100.0

            # compute midpoint in watts (kickr) and the requested y: kickr * (1 + weighted_avg_diff/100)
            zone_stats['zone_mid_kickr'] = float(ftp) * zone_stats['zone_mid_frac']
            zone_stats['y_kickr_adj'] = zone_stats['zone_mid_kickr'] * (1.0 + zone_stats['weighted_avg_diff'] / 100.0)

            # --- NEW: produce a range plot per zone (no histogram/bar chart) ---
            # For each original row in 'g', compute adjusted kickr = kickr * (1 + %diff/100)
            g['adjusted_kickr'] = g['kickr'].astype(float) * (1.0 + g['%diff'].astype(float) / 100.0)

            # compute per-zone min/max of adjusted kickr
            zone_ranges = (
                g.groupby('zone')['adjusted_kickr']
                .agg(['min', 'max', 'count'])
                .reset_index()
            )

            if zone_ranges.empty:
                print({'info': 'no adjusted kickr values per zone; skipping range plot'})
            else:
                # Prepare plotting positions and labels
                # Preserve the original zones order (do not sort alphabetically)
                ordered_zones = [z for z in zones.keys() if z in zone_ranges['zone'].values]
                if not ordered_zones:
                    print({'info': 'no matching zones found in zone_ranges; skipping range plot'})
                else:
                    # Reindex zone_ranges to the ordered zones
                    zone_ranges = zone_ranges.set_index('zone').loc[ordered_zones].reset_index()
                    labels = zone_ranges['zone'].astype(str).tolist()
                    x_pos = np.arange(len(labels))

                    # compute zone start/end in watts for label/context (preserve order)
                    zone_bounds = []
                    for z in labels:
                        lo_pct, hi_pct = zones.get(z, (None, None))
                        if lo_pct is None:
                            zone_bounds.append((np.nan, np.nan))
                        else:
                            start_w = float(ftp) * (lo_pct / 100.0)
                            end_w = float(ftp) * (hi_pct / 100.0)
                            zone_bounds.append((start_w, end_w))

                    # extract min/max arrays in the desired order
                    min_vals = zone_ranges['min'].astype(float).to_numpy()
                    max_vals = zone_ranges['max'].astype(float).to_numpy()

                    # Create a visual table summarizing zones instead of a line plot
                    try:
                        # Build a summary table per zone
                        # Map weighted_avg_diff from zone_stats if available
                        wadiff_map = {}
                        if 'zone' in zone_stats.columns and 'weighted_avg_diff' in zone_stats.columns:
                            wadiff_map = zone_stats.set_index('zone')['weighted_avg_diff'].to_dict()

                        table_rows = []
                        for lbl, (start_w, end_w), mn, mx in zip(labels, zone_bounds, min_vals, max_vals):
                            start_str = '' if pd.isna(start_w) else f"{int(round(start_w))}"
                            end_str = '' if pd.isna(end_w) else f"{int(round(end_w))}"
                            mn_str = '' if pd.isna(mn) else f"{int(round(mn))}"
                            mx_str = '' if pd.isna(mx) else f"{int(round(mx))}"
                            wad = wadiff_map.get(lbl, None)
                            if not pd.isna(wad) and wad is not None:
                                wad_str = f"{wad:.1f}%"
                            else:
                                wad_str = ''
                            # midpoint of adjusted values (mid_adj) = average of min and max adjusted_kickr
                            if not pd.isna(mn) and not pd.isna(mx):
                                mid_adj = int(round((float(mn) + float(mx)) / 2.0))
                            else:
                                mid_adj = ''

                            table_rows.append([lbl, f"{start_str}", f"{end_str}", f"{mn_str}", f"{mx_str}", f"{mid_adj}", wad_str])

                        cols = ['Zone', 'Start (W)', 'End (W)', 'Min adj (W)', 'Max adj (W)', 'Mid adj (W)', 'Weighted %diff']
                        table_df = pd.DataFrame(table_rows, columns=cols)

                        # Create figure sized based on number of rows
                        row_count = max(1, len(table_df))
                        fig_h = 0.8 + 0.4 * row_count
                        fig, ax = plt.subplots(figsize=(12, max(3, int(math.ceil(fig_h)))))

                        ax.axis('off')

                        # Create the table
                        tbl = ax.table(cellText=table_df.values,
                                       colLabels=table_df.columns,
                                       cellLoc='center',
                                       loc='center')
                        tbl.auto_set_font_size(False)
                        tbl.set_fontsize(10)
                        tbl.scale(1, 1.2)

                        # Dynamically size the Axes containing the table according to
                        # how many rows we have so the suptitle can be placed just
                        # above the table without overlapping.
                        try:
                            # row_count already computed above
                            # estimate height fraction: base + per-row increment
                            base = 0.12
                            per_row = 0.08
                            height_frac = min(0.88, base + per_row * row_count)
                            bottom_frac = 0.06
                            # ensure there's at least a small bottom margin
                            ax.set_position([0.05, bottom_frac, 0.9, height_frac])

                            # place the suptitle directly above the Axes
                            suptitle_y = bottom_frac + height_frac + 0.02
                            try:
                                fig.suptitle(f'Zones overzicht — FTP = {int(ftp)}', fontsize=14, y=suptitle_y)
                            except Exception:
                                fig.suptitle(f'Zones overzicht — FTP = {ftp}', fontsize=14, y=suptitle_y)
                        except Exception:
                            # fallback: simple suptitle at a fixed y
                            try:
                                fig.suptitle(f'Zones overzicht — FTP = {int(ftp)}', fontsize=14, y=0.96)
                            except Exception:
                                fig.suptitle(f'Zones overzicht — FTP = {ftp}', fontsize=14, y=0.96)

                        plot_path = PLOT_DIR / 'zones_tbl.png'
                        fig.savefig(plot_path, dpi=400)
                        plt.close(fig)
                        print({'info': 'Wrote zones visual table', 'plot': str(plot_path)})
                    except Exception as e:
                        print({'error': f'failed to create zones table plot: {e}'})

                    # x-tick labels: zone name + start-end watts (keep for downstream plots)
                    xticks_labels = []
                    for lbl, (start_w, end_w) in zip(labels, zone_bounds):
                        if pd.isna(start_w) or pd.isna(end_w):
                            xticks_labels.append(lbl)
                        else:
                            xticks_labels.append(f"{lbl}\n{int(start_w)}-{int(end_w)}W")


                    # also produce the watts-diff line plot (min/max vs midpoint)
                    zone_ranges['zone_mid_kickr'] = [ (b[0] + b[1]) / 2.0 if not pd.isna(b[0]) else np.nan for b in zone_bounds ]
                    min_wdiff = zone_ranges['min'] - zone_ranges['zone_mid_kickr']
                    max_wdiff = zone_ranges['max'] - zone_ranges['zone_mid_kickr']
