import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import List
from types import SimpleNamespace
from pathlib import Path
from configparser import ConfigParser, ExtendedInterpolation
from dcor import partial_distance_correlation, distance_correlation
from dcor.independence import distance_correlation_t_test
from matplotlib.ticker import FormatStrFormatter
from statsmodels.tsa.stattools import grangercausalitytests


day_width = mdates.datestr2num("2019-09-17 00:00:00") - mdates.datestr2num("2019-09-16 00:00:00")


def get_pref_id(pref_path: Path, target_pref: List[str]) -> List[int]:
    jp_pref = pd.read_csv(pref_path, index_col=2)
    if target_pref is None or len(target_pref) == 0:
        return (jp_pref['prefecture_code'] - 1).values.tolist()
    else:
        return (jp_pref.loc[target_pref]['prefecture_code'] - 1).values.tolist()


def get_twitter(
        twitter_path: Path, data_start_date: str, data_end_date: str, area_list: List[int]
) -> np.ndarray:
    twitter = pd.read_csv(twitter_path, index_col=0)
    twitter = twitter[area_list]
    twitter = twitter.loc[data_start_date:data_end_date]
    return twitter.values


def get_flow(flow_type: str, flow_path: Path, start_index: int, end_index: int, area_index: List[int]) -> np.ndarray:
    assert flow_type in str(flow_path), 'Please check if the flow data is compatible with flow type.'
    flow = np.load(flow_path)
    flow_pad = np.zeros((flow.shape[0], flow.shape[1] + 1))
    flow_pad[:, :flow.shape[1]] = flow
    flow_pad = flow_pad[start_index:end_index + 1, :]
    flow_pad = flow_pad[:, area_index]
    return flow_pad


def visualize_partial_distance_correlation(
        args: SimpleNamespace, ax: plt.Axes, flow_data, tweet_data, event_date_index, zoom_range,
        show_y, show_twiny, twiny_range, y_range, y_ticks, twiny_ticks
):
    zoom_range = range(24, zoom_range + args.zoom_step, args.zoom_step)
    ax2 = ax.twinx()
    for area_idx, area in enumerate(args.visualize_areas):
        pcorrs, corrs = list(), list()
        for zoom_offset in zoom_range:
            x, y, z = list(), list(), list()
            for d in event_date_index:
                fdata = flow_data[d - zoom_offset:d + 1, area_idx]
                tdata = tweet_data[d - zoom_offset:d + 1, area_idx]
                for start_idx in range(len(fdata) - args.window_size * 2):
                    x.append(tdata[start_idx:start_idx + args.window_size])
                    y.append(fdata[start_idx + args.window_size:start_idx + args.window_size * 2])
                    z.append(fdata[start_idx:start_idx + args.window_size])

            pcorr = partial_distance_correlation(x, y, z)
            corr = distance_correlation(np.hstack((x, z)), np.array(y))
            ttest = distance_correlation_t_test(np.array(x), np.array(y))
            # test_result = grangercausalitytests(np.hstack((x, z)), maxlag=24, verbose=False)
            # test_result2 = grangercausalitytests(np.array(x), maxlag=24, verbose=False)
            print(pcorr, corr, area, zoom_offset, ttest)
            pcorrs.append(pcorr)
            corrs.append(corr)

        ax.plot(zoom_range, pcorrs, lw=args.lw, label='Partial DisCorr')
        ax.get_yaxis().set_visible(show_y)
        ax.set_ylim(y_range)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.plot(zoom_range, corrs, '--', lw=args.lw, label='DisCorr', alpha=0.5)
        ax2.get_yaxis().set_visible(show_twiny)
        ax2.set_ylim(twiny_range)
        ax.grid(which='major', zorder=1)
        ax2.grid(which='major', zorder=1)
        ax.set_yticks(y_ticks)
        ax2.set_yticks(twiny_ticks)
        ax2.tick_params(axis='y', labelsize=8)
        ax.set_xticks(zoom_range[::4])
        ax.set_xticklabels(map(lambda x: str(x // 24), zoom_range[::4]))
        return ax, ax2


def visualize_tweet_count_effectiveness(args: SimpleNamespace) -> None:
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(
        start=args.flow_start_date, end=args.flow_end_date, freq=args.input_freq
    )]
    area_index = get_pref_id(args.prefecture_filepath, args.visualize_areas)

    typhoon_start_index = flow_all_times.index(args.typhoon_twitter_start_date)
    typhoon_end_index = flow_all_times.index(args.typhoon_twitter_end_date)
    typhoon_inflow = get_flow('inflow', args.inflow_filepath, typhoon_start_index, typhoon_end_index, area_index)
    typhoon_outflow = get_flow('outflow', args.outflow_filepath, typhoon_start_index, typhoon_end_index, area_index)
    typhoon_tweet = get_twitter(
        args.typhoon_tweet_filepath, args.typhoon_twitter_start_date, args.typhoon_twitter_end_date, args.visualize_areas
    )
    typhoon_event_dates_index = [flow_all_times.index(d) - typhoon_start_index for d in args.typhoon_event_dates]

    covid_start_index = flow_all_times.index(args.covid_twitter_start_date)
    covid_end_index = flow_all_times.index(args.covid_twitter_end_date)
    covid_inflow = get_flow('inflow', args.inflow_filepath, covid_start_index, covid_end_index, area_index)
    covid_outflow = get_flow('outflow', args.outflow_filepath, covid_start_index, covid_end_index, area_index)
    covid_tweet = get_twitter(
        args.covid_tweet_filepath, args.covid_twitter_start_date, args.covid_twitter_end_date, args.visualize_areas
    )
    covid_event_dates_index = [flow_all_times.index(d) - covid_start_index for d in args.covid_event_dates]

    fig = plt.figure(figsize=(args.figure_width, args.figure_height), dpi=args.figure_dpi)
    axes = fig.subplots(1, 2, gridspec_kw={'wspace': 0.1})
    visualize_partial_distance_correlation(
        args, axes[0], typhoon_inflow, typhoon_tweet, typhoon_event_dates_index, args.typhoon_zoom_range, True, False,
        [0.55, 0.855], [0, 0.61], [0.1, 0.3, 0.5], [0.6, 0.7, 0.8]
    )
    ax1, ax2 = visualize_partial_distance_correlation(
        args, axes[1], covid_outflow, covid_tweet, covid_event_dates_index, args.covid_zoom_range, False, True,
        [0.54, 0.91], [-0.01, 0.36], [0.05, 0.15, 0.25], [0.6, 0.7, 0.8]
    )
    axes[0].set_title('Typhoon Inflow', fontsize=12)
    axes[1].set_title('COVID-19 Outflow', fontsize=12)
    axes[0].tick_params(axis='x', length=0)
    axes[1].tick_params(axis='x', length=0)
    for ax in [axes[0], axes[1]]:
        ax.tick_params(axis='both', labelsize=8)
    fig.text(
        0.5, -0.03, 'Days Before Events',
        {'ha': 'center', 'va': 'bottom', 'rotation': 0, 'fontsize': 8}
    )
    fig.text(
        0.03, 0.5, 'Partial Distance Correlation\n(Partial DCorr)',
        {'ha': 'center', 'va': 'center', 'rotation': 90, 'rotation_mode': 'anchor', 'fontsize': 8}
    )
    fig.text(
        0.98, 0.5, 'Distance Correlation\n(DCorr)',
        {'ha': 'center', 'va': 'center', 'rotation': 270, 'rotation_mode': 'anchor', 'fontsize': 8}
    )
    lhandles = [*ax1.get_legend_handles_labels()[0], *ax2.get_legend_handles_labels()[0]]
    fig.legend(handles=lhandles, loc='lower center', ncol=2, bbox_to_anchor=(0.5, -0.16), prop={'size': 8})

    if args.output_path is not None and args.output_path != "":
        fig.savefig(args.output_path, bbox_inches='tight')


if __name__ == '__main__':
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('src/IEEE_IJCAI2022_data_visualization/figure2/tweet_count_effectiveness_2plot.conf')

    args = SimpleNamespace()
    args.typhoon_tweet_filepath = Path(config.get('DataPath', 'typhoon_tweet_filepath'))
    args.covid_tweet_filepath = Path(config.get('DataPath', 'covid_tweet_filepath'))
    args.inflow_filepath = Path(config.get('DataPath', 'inflow_filepath'))
    args.outflow_filepath = Path(config.get('DataPath', 'outflow_filepath'))
    args.prefecture_filepath = Path(config.get('DataPath', 'prefecture_filepath'))

    args.input_freq = config.get('DataDesc', 'input_freq')
    args.flow_start_date = config.get('DataDesc', 'flow_start_date')
    args.flow_end_date = config.get('DataDesc', 'flow_end_date')
    args.typhoon_twitter_start_date = config.get('DataDesc', 'typhoon_twitter_start_date')
    args.typhoon_twitter_end_date = config.get('DataDesc', 'typhoon_twitter_end_date')
    args.covid_twitter_start_date = config.get('DataDesc', 'covid_twitter_start_date')
    args.covid_twitter_end_date = config.get('DataDesc', 'covid_twitter_end_date')

    args.typhoon_event_dates = eval(config.get('AnalysisConfig', 'typhoon_event_dates'))
    args.covid_event_dates = eval(config.get('AnalysisConfig', 'covid_event_dates'))
    args.typhoon_zoom_range = config.getint('AnalysisConfig', 'typhoon_zoom_range')
    args.covid_zoom_range = config.getint('AnalysisConfig', 'covid_zoom_range')
    args.zoom_step = config.getint('AnalysisConfig', 'zoom_step')
    args.window_size = config.getint('AnalysisConfig', 'window_size')

    args.visualize_areas = eval(config.get('VisualizeConfig', 'visualize_areas'))
    args.figure_width = config.getint('VisualizeConfig', 'figure_width')
    args.figure_height = config.getint('VisualizeConfig', 'figure_height')
    args.figure_dpi = config.getint('VisualizeConfig', 'figure_dpi')
    args.lw = config.getfloat('VisualizeConfig', 'linewidth')

    args.output_path = config.get('Output', 'output_path')

    visualize_tweet_count_effectiveness(args)
