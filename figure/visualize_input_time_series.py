import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import matplotlib.patches as patches
from typing import List, Tuple
from types import SimpleNamespace
from pathlib import Path
from datetime import date
from configparser import ConfigParser, ExtendedInterpolation
from jpholiday import is_holiday
from PIL import Image


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


def norm_data(flow_data: np.ndarray, tweet_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return flow_data / np.max(flow_data, axis=0), tweet_data / np.max(tweet_data, axis=0)


def select_data(
        flow_data: np.ndarray, tweet_data: np.ndarray,
        data_start_date: str, data_end_date: str, freq: str,
        target_start_date: str, target_end_date: str
) -> Tuple[np.ndarray, np.ndarray]:
    twitter_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(
        start=data_start_date, end=data_end_date, freq=freq
    )]
    start_index = twitter_all_times.index(target_start_date)
    end_index = twitter_all_times.index(target_end_date)
    return flow_data[start_index:end_index + 1], tweet_data[start_index:end_index + 1]


def annotate_day_type(ax, ymin, ymax, date, holiday_color, weekday_color, weekend_color, zorder=0):
    if is_holiday(date):
        color = holiday_color
    elif date.weekday() in [0, 1, 2, 3, 4]:
        color = weekday_color
    else:
        color = weekend_color
    ax.add_patch(patches.Rectangle((
        mdates.datestr2num('{:d}-{:02d}-{:02d} 00:00:00'.format(date.year, date.month, date.day)), ymin),
        day_width, ymax - ymin, linewidth=0, facecolor=color, zorder=zorder
    ))


def visualize_typhoon_time_series(args: SimpleNamespace, flow_data: np.ndarray, tweet_data: np.ndarray) -> np.ndarray:
    # Preprocess data
    norm_flow_data, norm_tweet_data = norm_data(flow_data, tweet_data)
    fdata, tdata = select_data(
        norm_flow_data, norm_tweet_data, args.typhoon_twitter_start_date, args.typhoon_twitter_end_date,
        args.input_freq, args.visualize_typhoon_start_date, args.visualize_typhoon_end_date
    )
    xdata = pd.date_range(
        start=args.visualize_typhoon_start_date, end=args.visualize_typhoon_end_date, freq=args.visualize_typhoon_freq
    )

    # Generate plots
    fig = plt.figure(figsize=(args.figure_width, args.figure_height), dpi=args.figure_dpi)
    axes = fig.subplots(len(args.visualize_areas), 1, sharex=True, gridspec_kw={'hspace': 0.})
    for area_idx in range(len(args.visualize_areas)):
        ax = axes[area_idx]
        ax.plot(xdata, fdata[:, area_idx], args.flow_linecolor, lw=args.lw, label='Human Flow Volume ({})'.format(args.visualize_flow_type))
        ax.plot(xdata, tdata[:, area_idx], args.tweet_linecolor, lw=args.lw, label='Disaster-Related Tweet Count')

        # Format y-axis
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0., 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(args.visualize_areas[area_idx], fontweight='bold')
        ax.grid(which='major', zorder=1)

        # Annotate landfall time
        ax.vlines(mdates.datestr2num('2019-08-14 09:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)
        ax.vlines(mdates.datestr2num('2019-08-15 09:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)
        ax.vlines(mdates.datestr2num('2019-09-09 05:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)
        ax.vlines(mdates.datestr2num('2019-10-12 19:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)

        # Annotate Day Type
        for d in pd.date_range(start=args.visualize_typhoon_start_date, end=args.visualize_typhoon_end_date, freq='1D'):
            annotate_day_type(ax, -0.05, 1.05, d, args.holiday_facecolor, args.weekday_facecolor, args.weekend_facecolor)

    # Add y-label
    fig.supylabel(
        'Normalized Human {} and Tweet Count\nin 2019 Pacific Typhoon Season'.format(args.visualize_flow_type),
        fontweight='bold', ha='center'
    )

    # Add landfall labels
    fig.text(0.19, 0.976, 'Typhoon Krosa passed over')
    fig.text(0.45, 0.976, 'Typhoon Faxai made landfall')
    fig.text(0.78, 0.976, 'Typhoon Hagibis made landfall')

    # Format x-axis
    axes[-1].xaxis.set_major_locator(mdates.HourLocator(byhour=[0]))
    axes[-1].xaxis.set_minor_locator(mdates.HourLocator(byhour=[12]))
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter(''))
    axes[-1].xaxis.set_minor_formatter(mdates.DateFormatter('%m-%d'))
    plt.setp(axes[-1].xaxis.get_minorticklabels(), rotation=90)
    axes[-1].set_xlim([xdata[0], xdata[-1]])

    # Extract plot as numpy
    fig.tight_layout(pad=1)
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def visualize_covid_time_series(args: SimpleNamespace, flow_data: np.ndarray, tweet_data: np.ndarray) -> np.ndarray:
    # Preprocess data (Aggregate COVID data at day level for better visualize global temporal trending)
    norm_flow_data, norm_tweet_data = norm_data(flow_data, tweet_data)
    fdata, tdata = select_data(
        norm_flow_data, norm_tweet_data, args.covid_twitter_start_date, args.covid_twitter_end_date,
        args.input_freq, args.visualize_covid_start_date, args.visualize_covid_end_date
    )
    agg_flow_data = np.sum(np.reshape(fdata, (-1, 24, fdata.shape[1])), axis=1)
    agg_tweet_data = np.sum(np.reshape(tdata, (-1, 24, tdata.shape[1])), axis=1)
    fdata, tdata = norm_data(agg_flow_data, agg_tweet_data)
    xdata = pd.date_range(
        start=args.visualize_covid_start_date, end=args.visualize_covid_end_date, freq=args.visualize_covid_freq
    )

    # Generate plots
    fig = plt.figure(figsize=(args.figure_width, args.figure_height), dpi=args.figure_dpi)
    axes = fig.subplots(len(args.visualize_areas), 1, sharex=True, gridspec_kw={'hspace': 0.})
    for area_idx in range(len(args.visualize_areas)):
        ax = axes[area_idx]
        ax.plot(xdata, fdata[:, area_idx], args.flow_linecolor, lw=args.lw, label='Human Flow Volume ({})'.format(args.visualize_flow_type))
        ax.plot(xdata, tdata[:, area_idx], args.tweet_linecolor, lw=args.lw, label='Disaster-Related Tweet Count')

        # Format y-axis
        ax.set_ylim([-0.05, 1.05])
        ax.set_yticks([0., 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel(args.visualize_areas[area_idx], fontweight='bold')
        ax.grid(which='major', zorder=1)

        # Annotate Day Type
        for d in pd.date_range(start=args.visualize_covid_start_date, end=args.visualize_covid_end_date, freq='1D'):
            annotate_day_type(ax, -0.05, 1.05, d, args.holiday_facecolor, args.weekday_facecolor, args.weekend_facecolor)

        # Annotate the start time of first three waves of COVID-19 in Japan
        ax.vlines(mdates.datestr2num('2020-01-15 00:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)
        ax.vlines(mdates.datestr2num('2020-06-01 00:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)
        ax.vlines(mdates.datestr2num('2020-10-01 00:00:00'), -0.05, 1.05, colors='r', lw=1, linestyles='solid', zorder=4)

    # Add y-label
    fig.supylabel(
        'Normalized Human {} and Tweet Count\nduring COVID-19 Pandemic'.format(args.visualize_flow_type), fontweight='bold', ha='center'
    )

    # Add COVID-19 labels
    axes[-1].scatter([
        mdates.datestr2num('2020-04-11 12:00:00'),
        mdates.datestr2num('2020-08-07 12:00:00'),
        mdates.datestr2num('2021-01-08 12:00:00')
    ], [-0.03] * 3, 30, 'r', marker='^', label="Peak of Confirmed COVID-19 Cases", zorder=4)
    fig.text(0.19, 0.976, 'The first wave of COVID-19')
    fig.text(0.47, 0.976, 'The second wave of COVID-19')
    fig.text(0.77, 0.976, 'The third wave of COVID-19')

    # Format x-axis
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axes[-1].xaxis.set_minor_locator(mdates.DayLocator())
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%y-%m-%d'))
    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=90)
    axes[-1].set_xlim([xdata[0], xdata[-1]])

    # Add legend
    days = list()
    days.append(patches.Patch(edgecolor='k', facecolor=args.weekday_facecolor, label='Weekday'))
    if args.holiday_facecolor == args.weekend_facecolor:
        days.append(patches.Patch(edgecolor='k', facecolor=args.holiday_facecolor, label='Weekend or National Holiday'))
    else:
        days.append(patches.Patch(edgecolor='k', facecolor=args.holiday_facecolor, label='National Holiday'))
        days.append(patches.Patch(edgecolor='k', facecolor=args.weekend_facecolor, label='Weekend'))
    lhandles = [*axes[-1].get_legend_handles_labels()[0], *days]
    fig.legend(handles=lhandles, loc='lower center', ncol=6, bbox_to_anchor=(0.5, 0))

    # Extract plot as numpy
    fig.tight_layout(pad=1, rect=(0, 0.05, 1, 1))
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    return img.reshape(fig.canvas.get_width_height()[::-1] + (3,))


def visualize_input_time_series(args: SimpleNamespace) -> None:
    flow_all_times = [date.strftime('%Y-%m-%d %H:%M:%S') for date in pd.date_range(
        start=args.flow_start_date, end=args.flow_end_date, freq=args.input_freq
    )]
    area_index = get_pref_id(args.prefecture_filepath, args.visualize_areas)
    flow_type = args.visualize_flow_type.lower()
    flow_filepath = getattr(args, '{}_filepath'.format(flow_type))

    typhoon_start_index = flow_all_times.index(args.typhoon_twitter_start_date)
    typhoon_end_index = flow_all_times.index(args.typhoon_twitter_end_date)
    typhoon_flow = get_flow(flow_type, flow_filepath, typhoon_start_index, typhoon_end_index, area_index)
    typhoon_tweet = get_twitter(
        args.typhoon_tweet_filepath, args.typhoon_twitter_start_date, args.typhoon_twitter_end_date, args.visualize_areas
    )
    typhoon_img = visualize_typhoon_time_series(args, typhoon_flow, typhoon_tweet)

    covid_start_index = flow_all_times.index(args.covid_twitter_start_date)
    covid_end_index = flow_all_times.index(args.covid_twitter_end_date)
    covid_flow = get_flow(flow_type, flow_filepath, covid_start_index, covid_end_index, area_index)
    covid_tweet = get_twitter(
        args.covid_tweet_filepath, args.covid_twitter_start_date, args.covid_twitter_end_date, args.visualize_areas
    )
    covid_img = visualize_covid_time_series(args, covid_flow, covid_tweet)

    figure = Image.fromarray(np.concatenate((typhoon_img, covid_img), axis=0))
    figure.show()
    if args.output_path is not None and args.output_path != "":
        figure.save(args.output_path)


if __name__ == '__main__':
    config = ConfigParser(interpolation=ExtendedInterpolation())
    config.read('figures/src/figure1/visualize_input_time_series.conf')

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

    args.visualize_flow_type = config.get('VisualizeConfig', 'visualize_flow_type')
    args.visualize_areas = eval(config.get('VisualizeConfig', 'visualize_areas'))
    args.visualize_typhoon_start_date = config.get('VisualizeConfig', 'visualize_typhoon_start_date')
    args.visualize_typhoon_end_date = config.get('VisualizeConfig', 'visualize_typhoon_end_date')
    args.visualize_typhoon_freq = config.get('VisualizeConfig', 'visualize_typhoon_freq')
    args.visualize_covid_start_date = config.get('VisualizeConfig', 'visualize_covid_start_date')
    args.visualize_covid_end_date = config.get('VisualizeConfig', 'visualize_covid_end_date')
    args.visualize_covid_freq = config.get('VisualizeConfig', 'visualize_covid_freq')
    args.figure_width = config.getint('VisualizeConfig', 'figure_width')
    args.figure_height = config.getint('VisualizeConfig', 'figure_height')
    args.figure_dpi = config.getint('VisualizeConfig', 'figure_dpi')
    args.lw = config.getfloat('VisualizeConfig', 'linewidth')
    args.flow_linecolor = config.get('VisualizeConfig', 'flow_linecolor')
    args.tweet_linecolor = config.get('VisualizeConfig', 'tweet_linecolor')
    args.holiday_facecolor = config.get('VisualizeConfig', 'holiday_facecolor')
    args.weekday_facecolor = config.get('VisualizeConfig', 'weekday_facecolor')
    args.weekend_facecolor = config.get('VisualizeConfig', 'weekend_facecolor')

    args.output_path = config.get('Output', 'output_path')

    visualize_input_time_series(args)
