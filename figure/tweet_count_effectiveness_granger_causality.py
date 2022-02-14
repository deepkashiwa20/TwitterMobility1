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
from src.IEEE_IJCAI2022_data_visualization.figure1.visualize_input_time_series import norm_data


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


def visualize_granger_causality(
        args: SimpleNamespace, ax: plt.Axes, flow_data, tweet_data, window_size, lag, show_tick
):
    tests = list()
    for start_idx in range(0, len(flow_data), window_size):
        for area_idx, area in enumerate(args.visualize_areas):
            fdata = flow_data[start_idx:start_idx + window_size + 1, area_idx]
            tdata = tweet_data[start_idx:start_idx + window_size + 1, area_idx]
            if len(fdata) < 3 * 24 + 1:
                continue
            data = np.vstack((np.diff(tdata)[1:], np.diff(fdata)[1:])).transpose()
            tests.append(grangercausalitytests(data, maxlag=25))

    norm_flow_data, norm_tweet_data = norm_data(flow_data, tweet_data)
    agg_flow_data = np.sum(np.reshape(norm_flow_data, (-1, 24, norm_flow_data.shape[1])), axis=1)
    agg_tweet_data = np.sum(np.reshape(norm_tweet_data, (-1, 24, norm_tweet_data.shape[1])), axis=1)
    agg_flow_data, agg_tweet_data = norm_data(agg_flow_data, agg_tweet_data)
    ax.plot(range(0, len(flow_data), 24), agg_tweet_data, label='Tweet Count')
    ax2 = ax.twinx()
    ax2.plot(range(3*24+12, len(flow_data), window_size), [list(t.values())[lag][0]['ssr_ftest'][1] for t in tests], 'g', label='Granger Causality')
    ax.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax.set_ylim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    ax.grid(which='major', zorder=1)
    ax2.grid(which='major', zorder=1)
    if not show_tick:
        ax2.set_yticks([])
    # ax2.set_yticks([0, 0.16, 0.32, 0.48, 0.64, 0.8])


def visualize_autocorrelation(
        args: SimpleNamespace, ax: plt.Axes, flow_data, tweet_data, window_size, lag, show_tick
):
    tests = list()
    for start_idx in range(0, len(flow_data), window_size):
        for area_idx, area in enumerate(args.visualize_areas):
            fdata = flow_data[start_idx:start_idx + window_size + 1, area_idx]
            tdata = tweet_data[start_idx:start_idx + window_size + 1, area_idx]
            if len(fdata) < 3 * 24 + 1:
                continue
            tests.append(np.corrcoef(fdata, tdata)[0, 1])

    norm_flow_data, norm_tweet_data = norm_data(flow_data, tweet_data)
    agg_flow_data = np.sum(np.reshape(norm_flow_data, (-1, 24, norm_flow_data.shape[1])), axis=1)
    agg_tweet_data = np.sum(np.reshape(norm_tweet_data, (-1, 24, norm_tweet_data.shape[1])), axis=1)
    agg_flow_data, agg_tweet_data = norm_data(agg_flow_data, agg_tweet_data)
    ax.plot(range(0, len(flow_data), 24), agg_tweet_data, label='Tweet Count')
    ax2 = ax.twinx()
    ax2.plot(range(3*24+12, len(flow_data), window_size), tests, 'g', label='Auto-correlation')
    ax.tick_params(axis='both', labelsize=8)
    ax2.tick_params(axis='both', labelsize=8)
    ax.set_ylim([-0.1, 1.1])
    ax2.set_ylim([-0.1, 1.1])
    ax.grid(which='major', zorder=1)
    ax2.grid(which='major', zorder=1)
    if not show_tick:
        ax2.set_yticks([])


def visualize_tweet_count_effectiveness(args: SimpleNamespace) -> None:
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(
        start=args.flow_start_date, end=args.flow_end_date, freq=args.input_freq
    )]
    area_index = get_pref_id(args.prefecture_filepath, args.visualize_areas)

    typhoon_start_index = flow_all_times.index(args.typhoon_analysis_start_date)
    typhoon_end_index = flow_all_times.index(args.typhoon_analysis_end_date)
    typhoon_inflow = get_flow('inflow', args.inflow_filepath, typhoon_start_index, typhoon_end_index, area_index)
    typhoon_outflow = get_flow('outflow', args.outflow_filepath, typhoon_start_index, typhoon_end_index, area_index)
    typhoon_tweet = get_twitter(
        args.typhoon_tweet_filepath, args.typhoon_analysis_start_date, args.typhoon_analysis_end_date, args.visualize_areas
    )

    covid_start_index = flow_all_times.index(args.covid_analysis_start_date)
    covid_end_index = flow_all_times.index(args.covid_analysis_end_date)
    covid_inflow = get_flow('inflow', args.inflow_filepath, covid_start_index, covid_end_index, area_index)
    covid_outflow = get_flow('outflow', args.outflow_filepath, covid_start_index, covid_end_index, area_index)
    covid_tweet = get_twitter(
        args.covid_tweet_filepath, args.covid_analysis_start_date, args.covid_analysis_end_date, args.visualize_areas
    )

    fig = plt.figure(figsize=(args.figure_width, args.figure_height), dpi=args.figure_dpi)
    axes = fig.subplots(1, 2, sharey=True)
    visualize_granger_causality(
        args, axes[0], typhoon_outflow, typhoon_tweet, 24*7, args.typhoon_lag, False
    )
    visualize_granger_causality(
        args, axes[1], covid_outflow, covid_tweet, 24*7, args.covid_lag, True
    )

    if args.output_path is not None and args.output_path != "":
        fig.savefig(args.output_path, bbox_inches='tight')


if __name__ == '__main__':
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('src/IEEE_IJCAI2022_data_visualization/figure2/tweet_count_effectiveness_granger_causality.conf')

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

    args.typhoon_analysis_start_date = config.get('AnalysisConfig', 'typhoon_analysis_start_date')
    args.typhoon_analysis_end_date = config.get('AnalysisConfig', 'typhoon_analysis_end_date')
    args.covid_analysis_start_date = config.get('AnalysisConfig', 'covid_analysis_start_date')
    args.covid_analysis_end_date = config.get('AnalysisConfig', 'covid_analysis_end_date')
    args.typhoon_lag = config.getint('AnalysisConfig', 'typhoon_lag')
    args.covid_lag = config.getint('AnalysisConfig', 'covid_lag')

    args.visualize_areas = eval(config.get('VisualizeConfig', 'visualize_areas'))
    args.figure_width = config.getint('VisualizeConfig', 'figure_width')
    args.figure_height = config.getint('VisualizeConfig', 'figure_height')
    args.figure_dpi = config.getint('VisualizeConfig', 'figure_dpi')
    args.lw = config.getfloat('VisualizeConfig', 'linewidth')

    args.output_path = config.get('Output', 'output_path')

    visualize_tweet_count_effectiveness(args)
