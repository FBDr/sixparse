#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
#  comparse.py
#
#  Copyright 2017 Floris <floris@ndn-icarus-simulator>
#
#  This program is free software; you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation; either version 2 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software
#  Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston,
#  MA 02110-1301, USA.
#
#


import matplotlib.pyplot as plt
from matplotlib import container
import sys
import os
import seaborn as sns
import numpy as np, scipy.stats as st
import scipy as sp
import scipy.stats

sns.reset_orig()
plt.rcParams.update({'figure.max_open_warning': 0})
font = {'family': 'sans-serif',
        'weight': 'normal',
        'size': 7}

plt.rc('font', **font)


# Function from https://stackoverflow.com/questions/15033511/compute-a-confidence-interval-from-sample-data
def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    inter = se * sp.stats.t._ppf((1 + confidence) / 2., n - 1)
    return [m, inter]


def calc_nw_usage(tx_bytess, rx_bytess):
    nwu = ((rx_bytess + tx_bytess) / 1000) * 8  # IN kbits
    return nwu


def parse_run(idx, itrs, folder, parseNC):
    metricsnum_IP = 11
    metricsnum_icn = 18

    metrics_ip_x_n_2d = [[] for i in range(itrs)]
    metrics_ip_x_y_2d = [[] for i in range(itrs)]
    metrics_icn_ip_n_2d = [[] for i in range(itrs)]
    metrics_icn_icn_n_2d = [[] for i in range(itrs)]
    metrics_icn_ip_y_2d = [[] for i in range(itrs)]
    metrics_icn_icn_y_2d = [[] for i in range(itrs)]

    metrics_res_ip_x_n = [[] for i in range(metricsnum_IP)]
    metrics_res_ip_x_y = [[] for i in range(metricsnum_IP)]
    metrics_res_icn_ip_n = [[] for i in range(metricsnum_icn)]
    metrics_res_icn_icn_n = [[] for i in range(metricsnum_icn)]
    metrics_res_icn_ip_y = [[] for i in range(metricsnum_icn)]
    metrics_res_icn_icn_y = [[] for i in range(metricsnum_icn)]

    for itr in range(0, itrs):
        if parseNC:

            locs = [idx, idx + 2, idx + 3, idx + 1, idx + 4, idx + 5]
        else:
            locs = [0, 0, 0, idx, idx + 1, idx + 2]
        if parseNC:
            cur_path_ip = folder + "/RUN" + str(locs[0]) + "_" + str(itr) + "/results.txt"

            # IP no caching
            means = np.loadtxt(cur_path_ip, skiprows=1, usecols=(0,))
            means = list(means)
            metrics_ip_x_n_2d[itr] = means
            nw_us_cur = calc_nw_usage(means[4], means[5])
            metrics_ip_x_n_2d[itr].append(nw_us_cur)

            # ICN (IP) no caching
            cur_path_icn = folder + "/RUN" + str(locs[1]) + "_" + str(itr) + "/results.txt"
            means = np.loadtxt(cur_path_icn, skiprows=1, usecols=(0,))
            means = list(means)
            metrics_icn_ip_n_2d[itr] = means
            nw_us_cur = calc_nw_usage(means[4], means[5])
            metrics_icn_ip_n_2d[itr].append(nw_us_cur)
            # ICN (ICN) no caching
            cur_path_icn = folder + "/RUN" + str(locs[2]) + "_" + str(itr) + "/results.txt"

            means = np.loadtxt(cur_path_icn, skiprows=1, usecols=(0,))
            means = list(means)
            metrics_icn_icn_n_2d[itr] = means
            nw_us_cur = calc_nw_usage(means[4], means[5])
            metrics_icn_icn_n_2d[itr].append(nw_us_cur)

            metrics_ip_x_n_2d_flipped = map(list, zip(*metrics_ip_x_n_2d))
            metrics_icn_ip_n_2d_flipped = map(list, zip(*metrics_icn_ip_n_2d))
            metrics_icn_icn_n_2d_flipped = map(list, zip(*metrics_icn_icn_n_2d))
            for jdx in range(0, len(metrics_ip_x_n_2d_flipped)):
                metrics_res_ip_x_n[jdx] = mean_confidence_interval(metrics_ip_x_n_2d_flipped[jdx], 0.95)
            for jdx in range(0, len(metrics_icn_ip_n_2d_flipped)):
                metrics_res_icn_ip_n[jdx] = mean_confidence_interval(metrics_icn_ip_n_2d_flipped[jdx], 0.95)
                metrics_res_icn_icn_n[jdx] = mean_confidence_interval(metrics_icn_icn_n_2d_flipped[jdx], 0.95)

        # IP with caching
        cur_path_ip = folder + "/RUN" + str(locs[3]) + "_" + str(itr) + "/results.txt"
        means = np.loadtxt(cur_path_ip, skiprows=1, usecols=(0,))
        means = list(means)
        metrics_ip_x_y_2d[itr] = means
        nw_us_cur = calc_nw_usage(means[4], means[5])
        metrics_ip_x_y_2d[itr].append(nw_us_cur)

        # ICN (IP) with caching
        cur_path_icn = folder + "/RUN" + str(locs[4]) + "_" + str(itr) + "/results.txt"
        means = np.loadtxt(cur_path_icn, skiprows=1, usecols=(0,))
        means = list(means)
        metrics_icn_ip_y_2d[itr] = means
        nw_us_cur = calc_nw_usage(means[4], means[5])
        metrics_icn_ip_y_2d[itr].append(nw_us_cur)

        # ICN (ICN) with caching
        cur_path_icn = folder + "/RUN" + str(locs[5]) + "_" + str(itr) + "/results.txt"
        means = np.loadtxt(cur_path_icn, skiprows=1, usecols=(0,))
        means = list(means)
        metrics_icn_icn_y_2d[itr] = means
        nw_us_cur = calc_nw_usage(means[4], means[5])
        metrics_icn_icn_y_2d[itr].append(nw_us_cur)

    text_file_ip = open(cur_path_ip, "r")
    text_file_icn = open(cur_path_icn, "r")
    legend_ip = text_file_ip.readline().split()
    legend_icn = text_file_icn.readline().split()
    legend_ip.append("nwu")
    legend_icn.append("nwu")
    metrics_ip_x_y_2d_flipped = map(list, zip(*metrics_ip_x_y_2d))
    metrics_icn_ip_y_2d_flipped = map(list, zip(*metrics_icn_ip_y_2d))
    metrics_icn_icn_y_2d_flipped = map(list, zip(*metrics_icn_icn_y_2d))

    for jdx in range(0, len(metrics_ip_x_y_2d_flipped)):
        metrics_res_ip_x_y[jdx] = mean_confidence_interval(metrics_ip_x_y_2d_flipped[jdx], 0.95)

    for jdx in range(0, len(metrics_icn_ip_y_2d_flipped)):
        metrics_res_icn_ip_y[jdx] = mean_confidence_interval(metrics_icn_ip_y_2d_flipped[jdx], 0.95)
        metrics_res_icn_icn_y[jdx] = mean_confidence_interval(metrics_icn_icn_y_2d_flipped[jdx], 0.95)

    return_results = [dict(zip(legend_ip, metrics_res_ip_x_n)),
                      dict(zip(legend_ip, metrics_res_ip_x_y)),
                      dict(zip(legend_icn, metrics_res_icn_ip_n)),
                      dict(zip(legend_icn, metrics_res_icn_icn_n)),
                      dict(zip(legend_icn, metrics_res_icn_ip_y)),
                      dict(zip(legend_icn, metrics_res_icn_icn_y))]

    return return_results


def load_base_line_raw(folder, itr, trunc_time):
    metrics = 2
    raw_matrix_IP_3D = [[[] for i in range(metrics)] for j in range(2)]
    raw_matrix_ICN_3D = [[[] for i in range(metrics)] for j in range(4)]

    for cnt in range(1, 3):
        raw_matrix_IP_3D[cnt - 1] = create_raw_ip(folder, cnt, itr, trunc_time)  # IP no cache
        raw_matrix_IP_3D[cnt - 1] = create_raw_ip(folder, cnt, itr, trunc_time)  # IP with cache

    for cnt in range(1, 5):
        raw_matrix_ICN_3D[cnt - 1] = create_raw_icn(folder, 2 + cnt, itr, trunc_time)  # ICN(IP) no cache
    return [raw_matrix_IP_3D, raw_matrix_ICN_3D]


def create_raw_ip(folder, idx, itr, trunc_time):
    raw_metrics_ip = 2  # Hops and delay
    return_delay = [[] for j in range(0, itr)]
    return_hops = [[] for j in range(0, itr)]
    global exclude_itr

    for itrx in range(0, itr):
        if itrx not in exclude_itr:
            cur_path = folder + "/RUN" + str(idx) + "_" + str(itrx) + "/hopdelay.txt"
            if ("SH" in folder):
                time, hops, delay = np.loadtxt(cur_path, usecols=(1, 2, 3), unpack=True)
            else:
                time, hops, delay = np.loadtxt(cur_path, usecols=(1, 3, 4), unpack=True)

            hop_f = []
            for hop_i, delay_i in zip(hops, delay):
                if delay_i == 0:
                    hop_f.append(hop_i)
                else:
                    hop_f.append(hop_i + 1)

            hops = hop_f
            start_index = next(i for i, cvalue in enumerate(time) if cvalue > trunc_time)
            return_delay[itrx] = delay[start_index:]
            return_delay[itrx] = [it for it in return_delay[itrx] if it < 100]  # Remove NS delays
            return_hops[itrx] = hops[start_index:]

    return_hops = [j for i in return_hops for j in i]
    return_delay = [j for i in return_delay for j in i]
    return_matrix = [return_hops, return_delay]
    return return_matrix


def create_raw_icn(folder, idxxx, itr, trunc_time):
    raw_metrics_ip = 2  # Hops and delay
    trunc_time = trunc_time
    return_delay = [[] for j in range(0, itr)]
    return_hops = [[] for j in range(0, itr)]

    global exclude_itr

    for itrx in range(0, itr):
        if itrx not in exclude_itr:
            cur_path = folder + "/RUN" + str(idxxx) + "_" + str(itrx) + "/app-delays-trace.txt"
            time, delay, hops, = np.loadtxt(cur_path, skiprows=1,
                                            usecols=(0, 5, 8),
                                            unpack=True)
            delay_filtered = []
            hops_filtered = []
            delay_filtered_mult = []
            jdx = 0
            for delay_i, hop_i, time_i in zip(delay, hops, time):
                jdx += 1
                if jdx % 2 and time_i > trunc_time and delay_i < 0.1:
                    delay_filtered.append(delay_i)
                    hops_filtered.append(hop_i)
            delay_filtered_mult = [x * 1000 for x in delay_filtered]
            return_delay[itrx] = delay_filtered_mult
            return_hops[itrx] = hops_filtered

    return_hops = [j for i in return_hops for j in i]
    return_delay = [j for i in return_delay for j in i]

    return_matrix = [return_hops, return_delay]
    return return_matrix


def plot_sensitivity_line(gtitle, folder, metric, bse_val, res_dict_ip_x_n, res_dict_ip_x_y, res_dict_icn_ip_n,
                          res_dict_icn_icn_n,
                          res_dict_icn_ip_y, res_dict_icn_icn_y, minrun, maxrun, labels, title, ylabel, useNCruns,
                          x_array_c, use_scale, x_unit, y_unit, simtime, cnr, line):
    lgd = 0
    fag = 0
    if simtime:
        divisor = simtime
    else:
        divisor = 1
    data_point_ip_x_n = []
    data_point_ip_x_y = []
    data_point_icn_ip_n = []
    data_point_icn_icn_n = []
    data_point_icn_ip_y = []
    data_point_icn_icn_y = []

    error_point_ip_x_n = []
    error_point_ip_x_y = []
    error_point_icn_ip_n = []
    error_point_icn_icn_n = []
    error_point_icn_ip_y = []
    error_point_icn_icn_y = []
    x_array = list(x_array_c)

    global fgn
    global subn
    # plt.figure(fgn)
    plt.subplot(2, 3, subn + 1)
    for jdx in range(minrun, maxrun):
        if useNCruns == 'true':
            data_point_ip_x_n.append(res_dict_ip_x_n[jdx][metric][0] / divisor)
            data_point_ip_x_y.append(res_dict_ip_x_y[jdx][metric][0] / divisor)
            data_point_icn_ip_n.append(res_dict_icn_ip_n[jdx][metric][0] / divisor)
            data_point_icn_icn_n.append(res_dict_icn_icn_n[jdx][metric][0] / divisor)
            data_point_icn_ip_y.append(res_dict_icn_ip_y[jdx][metric][0] / divisor)
            data_point_icn_icn_y.append(res_dict_icn_icn_y[jdx][metric][0] / divisor)

            error_point_ip_x_n.append(res_dict_ip_x_n[jdx][metric][1])
            error_point_ip_x_y.append(res_dict_ip_x_y[jdx][metric][1])
            error_point_icn_ip_n.append(res_dict_icn_ip_n[jdx][metric][1])
            error_point_icn_icn_n.append(res_dict_icn_icn_n[jdx][metric][1])
            error_point_icn_ip_y.append(res_dict_icn_ip_y[jdx][metric][1])
            error_point_icn_icn_y.append(res_dict_icn_icn_y[jdx][metric][1])
        else:
            data_point_ip_x_y.append(res_dict_ip_x_y[jdx][metric][0] / divisor)
            data_point_icn_ip_y.append(res_dict_icn_ip_y[jdx][metric][0] / divisor)
            data_point_icn_icn_y.append(res_dict_icn_icn_y[jdx][metric][0] / divisor)

            error_point_ip_x_y.append(res_dict_ip_x_y[jdx][metric][1])
            error_point_icn_ip_y.append(res_dict_icn_ip_y[jdx][metric][1])
            error_point_icn_icn_y.append(res_dict_icn_icn_y[jdx][metric][1])

    if useNCruns == 'false':
        for iterator in range(len(data_point_ip_x_y)):
            data_point_ip_x_n.append(res_dict_ip_x_n[0][metric][0] / divisor)
            data_point_icn_ip_n.append(res_dict_icn_ip_n[0][metric][0] / divisor)
            data_point_icn_icn_n.append(res_dict_icn_icn_n[0][metric][0] / divisor)

            error_point_ip_x_n.append(res_dict_ip_x_n[0][metric][1])
            error_point_icn_ip_n.append(res_dict_icn_ip_n[0][metric][1])
            error_point_icn_icn_n.append(res_dict_icn_icn_n[0][metric][1])

    N = np.arange(len(data_point_ip_x_y))
    if not use_scale:
        x_array = N
    dist = 0.03
    alphas = 0.9
    msize = 5
    linew = 1.2
    if cnr:
        data_point_ip_x_n = [data_point_ip_x_n[0]] * len(data_point_ip_x_n)
        data_point_icn_icn_n = [data_point_icn_icn_n[0]] * len(data_point_ip_x_n)
        data_point_icn_ip_n = [data_point_icn_ip_n[0]] * len(data_point_ip_x_n)

        error_point_ip_x_n = [error_point_ip_x_n[0]] * len(data_point_ip_x_n)
        error_point_icn_icn_n = [error_point_icn_icn_n[0]] * len(data_point_ip_x_n)
        error_point_icn_ip_n = [error_point_icn_ip_n[0]] * len(data_point_ip_x_n)



    # # Add baseline
    # if (use_scale) and (bse_val != 0):
    #     x_array.append(bse_val)
    #     data_point_ip_x_n.append(res_dict_ip_x_n[0][metric][0] / divisor)
    #     data_point_ip_x_y.append(res_dict_ip_x_y[0][metric][0] / divisor)
    #     data_point_icn_ip_n.append(res_dict_icn_ip_n[0][metric][0] / divisor)
    #     data_point_icn_icn_n.append(res_dict_icn_icn_n[0][metric][0] / divisor)
    #     data_point_icn_ip_y.append(res_dict_icn_ip_y[0][metric][0] / divisor)
    #     data_point_icn_icn_y.append(res_dict_icn_icn_y[0][metric][0] / divisor)
    #
    #     error_point_ip_x_n.append(res_dict_ip_x_n[0][metric][1])
    #     error_point_ip_x_y.append(res_dict_ip_x_y[0][metric][1])
    #     error_point_icn_ip_n.append(res_dict_icn_ip_n[0][metric][1])
    #     error_point_icn_icn_n.append(res_dict_icn_icn_n[0][metric][1])
    #     error_point_icn_ip_y.append(res_dict_icn_ip_y[0][metric][1])
    #     error_point_icn_icn_y.append(res_dict_icn_icn_y[0][metric][1])
    #
    #     data_point_ip_x_n = [q for _, q in sorted(zip(x_array, data_point_ip_x_n))]
    #     data_point_ip_x_y = [q for _, q in sorted(zip(x_array, data_point_ip_x_y))]
    #     data_point_icn_ip_n = [q for _, q in sorted(zip(x_array, data_point_icn_ip_n))]
    #     data_point_icn_icn_n = [q for _, q in sorted(zip(x_array, data_point_icn_icn_n))]
    #     data_point_icn_ip_y = [q for _, q in sorted(zip(x_array, data_point_icn_ip_y))]
    #     data_point_icn_icn_y = [q for _, q in sorted(zip(x_array, data_point_icn_icn_y))]
    #
    #     error_point_ip_x_n = [q for _, q in sorted(zip(x_array, error_point_ip_x_n))]
    #     error_point_ip_x_y = [q for _, q in sorted(zip(x_array, error_point_ip_x_y))]
    #     error_point_icn_ip_n = [q for _, q in sorted(zip(x_array, error_point_icn_ip_n))]
    #     error_point_icn_icn_n = [q for _, q in sorted(zip(x_array, error_point_icn_icn_n))]
    #     error_point_icn_ip_y = [q for _, q in sorted(zip(x_array, error_point_icn_ip_y))]
    #     error_point_icn_icn_y = [q for _, q in sorted(zip(x_array, error_point_icn_icn_y))]
    #
    #     x_array = sorted(x_array)

    if (metric == "nwu"):
        error_point_ip_x_n = 0
        error_point_ip_x_y = 0
        error_point_icn_ip_n = 0
        error_point_icn_icn_n = 0
        error_point_icn_ip_y = 0
        error_point_icn_icn_y = 0

    plt.minorticks_on()
    if line:

        plt.errorbar(x_array, data_point_ip_x_n, yerr=error_point_ip_x_n, capsize=2, elinewidth=1, markeredgewidth=1,
                     marker='s',
                     label='IP - no caching', alpha=alphas, markersize=msize, lw=linew)
        plt.errorbar(x_array, data_point_ip_x_y, yerr=error_point_ip_x_y, capsize=2, elinewidth=1,
                     markeredgewidth=1,
                     marker='o', label='IP - with caching ', alpha=alphas, markersize=msize, lw=linew)
        plt.errorbar(x_array, data_point_icn_ip_n, yerr=error_point_icn_ip_n, capsize=2, elinewidth=1,
                     markeredgewidth=1, marker='H', label='ICN (IP) - no caching', alpha=alphas, markersize=msize,
                     lw=linew)
        plt.errorbar(x_array, data_point_icn_ip_y, yerr=error_point_icn_ip_y, capsize=2, elinewidth=1,
                     markeredgewidth=1,
                     marker='X', label='ICN (IP) - with caching', alpha=alphas, markersize=msize, lw=linew)
        plt.errorbar(x_array, data_point_icn_icn_n, yerr=error_point_icn_icn_n, capsize=2, elinewidth=1,
                     markeredgewidth=1,
                     marker='^', label='ICN (ICN) - no caching', alpha=alphas, markersize=msize, lw=linew)
        plt.errorbar(x_array, data_point_icn_icn_y, yerr=error_point_icn_icn_y, capsize=2, elinewidth=1,
                     markeredgewidth=1,
                     marker='P', label='ICN (ICN) - with caching', alpha=alphas, markersize=msize, lw=linew)
    else:
        ax = plt.gca()
        ax.tick_params(axis='x', which='minor', bottom='off')
        width = 0.12
        plt.bar(x_array - 2.5 * width, reorder_cns(data_point_ip_x_n), width, yerr=reorder_cns(error_point_ip_x_n),
                capsize=2,
                label='IP - no caching', alpha=alphas)
        plt.bar(x_array - 1.5 * width, reorder_cns(data_point_ip_x_y), width, yerr=reorder_cns(error_point_ip_x_y),
                capsize=2,
                label='IP - with caching ', alpha=alphas)
        plt.bar(x_array - 0.5 * width, reorder_cns(data_point_icn_ip_n), width, yerr=reorder_cns(error_point_icn_ip_n),
                capsize=2,
                label='ICN (IP) - no caching', alpha=alphas)
        plt.bar(x_array + 0.5 * width, reorder_cns(data_point_icn_ip_y), width, yerr=reorder_cns(error_point_icn_ip_y),
                capsize=2,
                label='ICN (IP) - with caching', alpha=alphas)
        plt.bar(x_array + 1.5 * width, reorder_cns(data_point_icn_icn_n), width,
                yerr=reorder_cns(error_point_icn_icn_n), capsize=2,
                label='ICN (ICN) - no caching', alpha=alphas)
        plt.bar(x_array + 2.5 * width, reorder_cns(data_point_icn_icn_y), width,
                yerr=reorder_cns(error_point_icn_icn_y), capsize=2,
                label='ICN (ICN) - with caching', alpha=alphas)
    if bse_val != 0:
        plt.axvline(x=bse_val, color='k', linewidth=alphas, linestyle='--', alpha=alphas, markersize=msize)
    plt.ylabel(y_unit)

    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.2)
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.2)
    if not use_scale:
        plt.xticks(N, labels)
    else:
        plt.xlabel(x_unit)
    plt.title(gtitle)

    fgn += 1
    subn += 1


def box_plot(folder, graphtitle, data):
    global fgn
    plt.figure(fgn)
    plt.title("Test")
    plt.boxplot(data)
    fgn += 1
    plt.tight_layout(pad=.0, w_pad=1.0, h_pad=1.0)
    filename = str(folder) + "/" + "Baseline" + str(graphtitle) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_inches='tight')


def reorder(data):
    data[4], data[3] = data[3], data[4]
    return data


def reorder_cns(data):
    if not isinstance(data, int):
        data[2], data[1] = data[1], data[2]
    return data


def violin_plot_delay(folder, graphtitle, data, res_dict_ip_x_n, res_dict_ip_x_y, res_dict_icn_ip_n, res_dict_icn_icn_n,
                      res_dict_icn_ip_y, res_dict_icn_icn_y):
    global fgn
    percentiles90 = []
    percentiles10 = []
    means = []
    plt.figure(fgn)
    # a, b = 3, 4
    data = reorder(data)
    data_points, conf_points = retrieve_base_data_from_bucket("delay", res_dict_ip_x_n,
                                                              res_dict_ip_x_y,
                                                              res_dict_icn_ip_n,
                                                              res_dict_icn_ip_y,
                                                              res_dict_icn_icn_n,
                                                              res_dict_icn_icn_y)
    N = np.arange(6)
    lbl = ["IP\nno caching", "IP\nwith caching", "ICN(IP)\nno caching", "ICN(IP)\nwith caching", "ICN(ICN)\nno caching",
           "ICN(ICN)\nwith caching"]

    for set in data:
        percentiles90.append(np.percentile(set, 90))
        percentiles10.append(np.percentile(set, 10))
    plt.figure(fgn)
    plt.title("End-to-end delay")

    g = sns.violinplot(data=data, inner=None, cut=0, color="skyblue", linewidth=0, gridsize=300)
    g.set_xticklabels(lbl)
    plt.ylim(-2, max(percentiles90) * 1.5)

    plt.plot(N, percentiles90, 'v', label="90th percentile")
    plt.errorbar(N, data_points, yerr=conf_points, capsize=2, elinewidth=1,
                 markeredgewidth=1,
                 fmt='o', label="Average")
    plt.plot(N, percentiles10, '^', label="10th percentile")
    plt.minorticks_on()
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.2)
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.2)
    plt.ylabel("Delay (ms)")
    lgd = plt.legend()
    fgn += 1
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    filename = str(folder) + "/" + "Baseline" + str(graphtitle) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_inches='tight')


def violin_plot_hops(folder, graphtitle, data):
    global fgn
    percentiles90 = []
    percentiles10 = []
    means = []
    plt.figure(fgn)
    ax = plt.gca()
    major_ticks = np.arange(0, 20, 2)
    ax.set_yticks(major_ticks)
    data = reorder(data)
    N = np.arange(6)
    lbl = ["IP\nno caching", "IP\nwith caching", "ICN(IP)\nno caching", "ICN(IP)\nwith caching", "ICN(ICN)\nno caching",
           "ICN(ICN)\nwith caching"]

    # data[b], data[a] = data[a], data[b]
    # lbl[b], lbl[a] = lbl[a], lbl[b]

    plt.title("Hop count")
    g = sns.violinplot(data=data, inner=None, cut=0, color="skyblue", linewidth=0)
    g.set_xticklabels(lbl)
    for set in data:
        percentiles90.append(np.percentile(set, 90))
        percentiles10.append(np.percentile(set, 10))
        means.append(np.mean(set))
    plt.plot(N, percentiles90, 'v', label="90th percentile")
    plt.plot(N, means, 'o', label="Average")
    plt.plot(N, percentiles10, '^', label="10th percentile")
    plt.minorticks_on()
    # Customize the major grid
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', alpha=0.2, )
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', alpha=0.2)
    plt.ylabel("Number of hops")
    fgn += 1
    lgd = plt.legend()
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    filename = str(folder) + "/" + "Baseline" + str(graphtitle) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_inches='tight')


def histograms(data):
    global fgn
    plt.figure(fgn)
    N = np.arange(6)
    lbl = ["IP", "IP - CoAP caching", "ICN(IP) no cache", "ICN(ICN) no cache", "ICN(IP) with cache",
           "ICN(ICN) with cache"]
    plt.xticks(N + 1, lbl)
    plt.title("Hop count")
    plt.hist(data[1], edgecolor='black', linewidth=1.2)
    fgn += 1


def bar_network_usage(folder, graphtitle, res_dict_ip_x_n, res_dict_ip_x_y, res_dict_icn_ip_n, res_dict_icn_icn_n,
                      res_dict_icn_ip_y, res_dict_icn_icn_y, divid):
    data_point = []
    conf_point = []
    data_point, conf_point = retrieve_base_data_from_bucket("nwu", res_dict_ip_x_n,
                                                            res_dict_ip_x_y,
                                                            res_dict_icn_ip_n,
                                                            res_dict_icn_ip_y,
                                                            res_dict_icn_icn_n,
                                                            res_dict_icn_icn_y)
    alphas = 0.9
    print data_point
    data_point = [x / divid for x in data_point]  # From kb to kbps
    global fgn
    plt.figure(fgn)
    lbl = ["IP\nno caching", "IP\nwith caching", "ICN(IP)\nno caching", "ICN(IP)\nwith caching", "ICN(ICN)\nno caching",
           "ICN(ICN)\nwith caching"]
    y_pos = np.arange(len(lbl))

    plt.minorticks_on()
    # Customize the major grid
    ax = plt.gca()
    ax.set_axisbelow(True)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', zorder=0, alpha=0.2)
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', zorder=0, alpha=0.2)
    # color = ['blue', 'blue', 'orange', 'orange', 'green', 'green']
    plt.bar(y_pos[0], data_point[0], align='center', alpha=alphas)  # ecolor='black', capsize=2,
    plt.bar(y_pos[1], data_point[1], align='center', alpha=alphas)  # ecolor='black', capsize=2,
    plt.bar(y_pos[2], data_point[2], align='center', alpha=alphas)  # ecolor='black', capsize=2,
    plt.bar(y_pos[3], data_point[3], align='center', alpha=alphas)  # ecolor='black', capsize=2,
    plt.bar(y_pos[4], data_point[4], align='center', alpha=alphas)  # ecolor='black', capsize=2,
    plt.bar(y_pos[5], data_point[5], align='center', alpha=alphas)  # ecolor='black', capsize=2,
    plt.xticks(y_pos, lbl)
    plt.ylabel('Average network usage (kb/s)')
    plt.title('Average network usage')

    fgn += 1
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    filename = str(folder) + "/" + "Baseline" + str(graphtitle) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_inches='tight')


def retrieve_base_data_from_bucket(metric, res_dict_1, res_dict_2, res_dict_3, res_dict_4,
                                   res_dict_5, res_dict_6):
    data_point = []
    conf_point = []
    data_point.append(res_dict_1[0][metric][0])
    data_point.append(res_dict_2[0][metric][0])
    data_point.append(res_dict_3[0][metric][0])
    data_point.append(res_dict_4[0][metric][0])
    data_point.append(res_dict_5[0][metric][0])
    data_point.append(res_dict_6[0][metric][0])

    conf_point.append(res_dict_1[0][metric][1])
    conf_point.append(res_dict_2[0][metric][1])
    conf_point.append(res_dict_3[0][metric][1])
    conf_point.append(res_dict_4[0][metric][1])
    conf_point.append(res_dict_5[0][metric][1])
    conf_point.append(res_dict_6[0][metric][1])
    return [data_point, conf_point]


def bar_cache_usage(folder, graphtitle, res_dict_ip_x_n, res_dict_ip_x_y, res_dict_icn_ip_n, res_dict_icn_icn_n,
                    res_dict_icn_ip_y, res_dict_icn_icn_y):
    data_point_mean = []
    conf_point_mean = []
    data_point_max = []
    conf_point_max = []
    data_point_mean.append(res_dict_ip_x_y[0]["average_cu"][0])

    data_point_mean.append(res_dict_icn_ip_y[0]["av_gtw"][0])
    data_point_mean.append(res_dict_icn_ip_y[0]["av_sensor"][0])
    data_point_mean.append(res_dict_icn_ip_y[0]["av_filtered_bh"][0])

    data_point_mean.append(res_dict_icn_icn_y[0]["av_gtw"][0])
    data_point_mean.append(res_dict_icn_icn_y[0]["av_sensor"][0])
    data_point_mean.append(res_dict_icn_icn_y[0]["av_filtered_bh"][0])
    conf_point_mean.append(res_dict_ip_x_y[0]["average_cu"][1])

    conf_point_mean.append(res_dict_icn_ip_y[0]["av_gtw"][1])
    conf_point_mean.append(res_dict_icn_ip_y[0]["av_sensor"][1])
    conf_point_mean.append(res_dict_icn_ip_y[0]["av_filtered_bh"][1])

    conf_point_mean.append(res_dict_icn_icn_y[0]["av_gtw"][1])
    conf_point_mean.append(res_dict_icn_icn_y[0]["av_sensor"][1])
    conf_point_mean.append(res_dict_icn_icn_y[0]["av_filtered_bh"][1])

    data_point_max.append(res_dict_ip_x_y[0]["max_cu"][0])

    data_point_max.append(res_dict_icn_ip_y[0]["max_gtw"][0])
    data_point_max.append(res_dict_icn_ip_y[0]["max_sensor"][0])
    data_point_max.append(res_dict_icn_ip_y[0]["max_bh"][0])

    data_point_max.append(res_dict_icn_icn_y[0]["max_gtw"][0])
    data_point_max.append(res_dict_icn_icn_y[0]["max_sensor"][0])
    data_point_max.append(res_dict_icn_icn_y[0]["max_bh"][0])

    conf_point_max.append(res_dict_ip_x_y[0]["max_cu"][1])

    conf_point_max.append(res_dict_icn_ip_y[0]["max_gtw"][1])
    conf_point_max.append(res_dict_icn_ip_y[0]["max_sensor"][1])
    conf_point_max.append(res_dict_icn_ip_y[0]["max_bh"][1])

    conf_point_max.append(res_dict_icn_icn_y[0]["max_gtw"][1])
    conf_point_max.append(res_dict_icn_icn_y[0]["max_sensor"][1])
    conf_point_max.append(res_dict_icn_icn_y[0]["max_bh"][1])

    global fgn
    plt.figure(fgn)
    lbl = ["IP\nwith caching", "ICN(IP)\nwith caching\n(gateway routers)", "ICN(IP)\nwith caching\n(IoT nodes)",
           "ICN(IP)\nwith caching\n(backhaul)", "ICN(ICN)\nwith caching\n(gateway routers)",
           "ICN(ICN)\nwith caching\n(IoT nodes)",
           "ICN(ICN)\nwith caching\n(backhaul)"]
    y_pos = np.arange(len(lbl))
    alphas = 0.9
    plt.bar(y_pos[0], 0, align='center')
    plt.bar(y_pos[0], data_point_mean[0], 0.5, align='center', yerr=conf_point_mean[0:1], ecolor='black', capsize=2,
            alpha=alphas)
    plt.bar(y_pos[0], 0, align='center')
    plt.bar(y_pos[1:4], data_point_mean[1:4], 0.5, align='center', yerr=conf_point_mean[1:4], ecolor='black', capsize=2,
            alpha=alphas)
    plt.bar(y_pos[0], 0, align='center')
    plt.bar(y_pos[4:7], data_point_mean[4:7], 0.5, align='center', yerr=conf_point_mean[4:7], ecolor='black', capsize=2,
            alpha=alphas)

    plt.plot(y_pos, data_point_max, 'h', label="Maximum cache usage")
    plt.xticks(y_pos, lbl)
    plt.ylabel('Percent')
    plt.title('Cache capacity usage')
    fgn += 1
    plt.legend()
    plt.minorticks_on()
    # Customize the major grid
    ax = plt.gca()
    ax.set_axisbelow(True)
    major_ticks = np.arange(0, 4.5, 0.5)
    ax.set_yticks(major_ticks)
    plt.grid(which='major', linestyle='-', linewidth='0.5', color='red', zorder=0, alpha=0.2)
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black', zorder=0, alpha=0.2)
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    filename = str(folder) + "/" + "Baseline" + str(graphtitle) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_inches='tight')


def pie_charts_cache(folder, graphtitle, res_dict_ip_x_n, res_dict_ip_x_y, res_dict_icn_ip_n, res_dict_icn_icn_n,
                     res_dict_icn_ip_y, res_dict_icn_icn_y):
    # PIEs
    global fgn
    ttl_h = 0.95
    fig = plt.figure(fgn)
    labels = ['From producer', 'From network cache']
    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])

    plt.subplot(2, 3, 1)
    plt.title("IP\nno caching", y=ttl_h)
    pie_chart_generator(res_dict_ip_x_n, True)

    plt.subplot(2, 3, 2)
    plt.title("IP\nwith caching", y=ttl_h)
    pie_chart_generator(res_dict_ip_x_y, False)

    plt.subplot(2, 3, 3)
    plt.title("ICN(IP)\nno caching", y=ttl_h)
    pie_chart_generator(res_dict_icn_ip_n, True)

    plt.subplot(2, 3, 4)
    plt.title("ICN(IP)\nwith caching", y=ttl_h)
    pie_chart_generator(res_dict_icn_ip_y, False)

    plt.subplot(2, 3, 5)
    plt.title("ICN(ICN)\nno caching", y=ttl_h)
    patches = pie_chart_generator(res_dict_icn_icn_n, True)
    plt.legend(patches, labels, loc='upper center', bbox_to_anchor=(0.5, 0))
    plt.subplot(2, 3, 6)
    plt.title("ICN(ICN)\nwith caching", y=ttl_h)
    pie_chart_generator(res_dict_icn_icn_y, False)

    fgn += 1
    plt.tight_layout(pad=1.0, w_pad=1.0, h_pad=1.0)
    filename = str(folder) + "/" + "Baseline" + str(graphtitle) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_inches='tight')


def pie_chart_generator(res_dict, no_cache):
    sizes = [res_dict[0]['from_producer'][0], res_dict[0]['from_cache'][0]]
    print res_dict[0]['from_producer'][1]
    print res_dict[0]['from_cache'][1]
    patches, texts, extra = plt.pie(sizes, shadow=False, startangle=90, autopct='%1.1f%%',
                                    wedgeprops={"edgecolor": "white", 'linewidth': 0.9, 'linestyle': 'solid',
                                                'antialiased': True})
    plt.axis('equal')
    return patches


def plot_sens(folder, title, labels, bse_val, res_dict_ip_x_n, res_dict_ip_x_y, res_dict_icn_ip_n, res_dict_icn_icn_n,
              res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, useNCruns, x_array, use_scale, x_unit, div_time,
              cnr, line):
    lab = labels
    global subn
    global fgn
    subn = 0
    plt.figure(fgn)
    print title
    plot_sensitivity_line("Average end-to-end delay", folder, 'delay', bse_val, res_dict_ip_x_n, res_dict_ip_x_y,
                          res_dict_icn_ip_n,
                          res_dict_icn_icn_n,
                          res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, lab, 1, 1, useNCruns, x_array,
                          use_scale, x_unit, 'Delay (ms)', 0, cnr, line)

    plot_sensitivity_line("Average hop count", folder, 'hops', bse_val, res_dict_ip_x_n, res_dict_ip_x_y,
                          res_dict_icn_ip_n,
                          res_dict_icn_icn_n,
                          res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, lab, 1, 1, useNCruns, x_array,
                          use_scale, x_unit, 'Number of hops', 0, cnr, line)

    plot_sensitivity_line("Average network usage", folder, 'nwu', bse_val, res_dict_ip_x_n, res_dict_ip_x_y,
                          res_dict_icn_ip_n,
                          res_dict_icn_icn_n,
                          res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, lab, 1, 1, useNCruns, x_array,
                          use_scale, x_unit, 'Average network usage (kb/s)', div_time, cnr, line)
    plot_sensitivity_line("Cache hit ratio", folder, 'from_cache', bse_val, res_dict_ip_x_n, res_dict_ip_x_y,
                          res_dict_icn_ip_n,
                          res_dict_icn_icn_n, res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, lab, 1, 1,
                          useNCruns, x_array, use_scale, x_unit, 'Requests satisfied by a cache (%)', 0, cnr, line)
    plot_sensitivity_line("Average cache\ncapacity usage", folder, 'average_cu', bse_val, res_dict_ip_x_n,
                          res_dict_ip_x_y, res_dict_icn_ip_n,
                          res_dict_icn_icn_n,
                          res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, lab, 1, 1, useNCruns, x_array,
                          use_scale, x_unit, ' Average cache usage (%)', 0, cnr, line)
    plot_sensitivity_line("Maximum cache\ncapacity usage", folder, 'max_cu', bse_val, res_dict_ip_x_n,
                          res_dict_ip_x_y, res_dict_icn_ip_n,
                          res_dict_icn_icn_n,
                          res_dict_icn_ip_y, res_dict_icn_icn_y, minRUN, maxRUN, lab, 1, 1, useNCruns, x_array,
                          use_scale, x_unit, 'Maximum cache utilisation (%)', 0, cnr, line)

    suptitle = plt.suptitle(title, fontsize=14, y=1.04)
    # lgd = plt.legend(loc=(1.1, 3))
    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    handles = [han[0] if isinstance(han, container.ErrorbarContainer) else han for han in handles]
    lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(-0.9, -0.3), ncol=3)
    plt.tight_layout(pad=0.5, w_pad=1, h_pad=2)
    filename = str(folder) + "/" + "Sensitivity" + str(title) + ".png"
    plt.savefig(filename.replace(" ", ""), dpi=300, bbox_extra_artists=(lgd, suptitle,), bbox_inches='tight', )


##
## Initialisation
## Initialisation
##
idx = 1  # Iterator for Runs
fgn = 0  # Figure number
subn = 0  # Subplot number
SH_cache_only_run_min = 85  # Range with only caching runs
SH_cache_only_run_max = 99  # Range with only caching runs

# Result dictionaries for every sub-run
sh_res_dict_ip_x_n = []
sh_res_dict_ip_x_y = []
sh_res_dict_icn_ip_n = []
sh_res_dict_icn_icn_n = []
sh_res_dict_icn_ip_y = []
sh_res_dict_icn_icn_y = []

fa_res_dict_ip_x_n = []
fa_res_dict_ip_x_y = []
fa_res_dict_icn_ip_n = []
fa_res_dict_icn_icn_n = []
fa_res_dict_icn_ip_y = []
fa_res_dict_icn_icn_y = []

res_container = []  # Temp container which holds result of current parsing round
box_data = [[]]  # Raw data for baseline scenario

## Read in RUNS and ITRS
runs_sh = int(sys.argv[1])
itrs_sh = int(sys.argv[2])
runs_fa = int(sys.argv[3])
itrs_fa = int(sys.argv[4])
exclude_itr = [4]  # Exclude iteration

# Directories
dir_sh = "Parsed_plots/SH"
dir_fa = "Parsed_plots/FA"
if not os.path.isdir(dir_sh):
    os.mkdir(dir_sh)
if not os.path.isdir(dir_fa):
    os.mkdir(dir_fa)
dir = "Parsed_plots"
path_sh = "SH2"
if runs_sh:
    # Smart Home Parsing
    #
    print "Parsing Smart Home..."
    while idx < runs_sh:
        if SH_cache_only_run_min <= idx <= SH_cache_only_run_max:
            sh_res_dict_ip_x_n.append(0)
            res_container = parse_run(idx, itrs_sh, path_sh, 0)
            sh_res_dict_ip_x_y.append(res_container[1])
            sh_res_dict_icn_ip_n.append(0)
            sh_res_dict_icn_icn_n.append(0)
            sh_res_dict_icn_ip_y.append(res_container[4])
            sh_res_dict_icn_icn_y.append(res_container[5])
            idx += 3
        else:
            res_container = parse_run(idx, itrs_sh, path_sh, 1)
            sh_res_dict_ip_x_n.append(res_container[0])
            sh_res_dict_ip_x_y.append(res_container[1])
            sh_res_dict_icn_ip_n.append(res_container[2])
            sh_res_dict_icn_icn_n.append(res_container[3])
            sh_res_dict_icn_ip_y.append(res_container[4])
            sh_res_dict_icn_icn_y.append(res_container[5])
            idx += 6

    # Sensitivity graphs
    lab = ["CL1", "CL2", "CL3"]
    x_label = range(3)
    plot_sens(dir_sh, "Consumer location", lab, 0, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n,
              sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 1, 4, 'true', x_label, 0, '', 50000, False, False)
    lab = ["1 min", "2 min", "5 min", "10 min", "20 min"]
    x_label = [1, 2, 5, 10, 20]
    plot_sens(dir_sh, "Data freshness period", lab, 15, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n,
              sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 4, 9, 'true', x_label, 1, 'Freshness setting (min)', 50000, True, True)
    x_label = [0, 0.5, 0.84, 1]
    lab = ["0", "0.5", "0.84", "1.0"]
    plot_sens(dir_sh, "Content popularity distribution", lab, 0.64, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y,
              sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n, sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 9, 13, 'true', x_label, 1, r'Zipf parameter ($\alpha$)', 50000, True, True)
    x_label = [0, 5, 10, 25, 50]
    lab = ["0 packets", "5 packets", "10 packets", "25 packets", "50 packets"]
    plot_sens(dir_sh, "Cache size", lab, 0, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n, sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 14, 19, 'false', x_label, 1, 'Cache size (Packets)', 50000, True, True)
    x_label = [0.5, 1, 10, 30, 60]
    lab = ["30 s", "1 min", "10 min", "30 min", "60 min"]
    plot_sens(dir_sh, "Request frequency", lab, 0, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n, sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 19, 24, 'true', x_label, 1, 'Transmission frequency (min)', 50000, False, True)
    x_label = [13, 18, 27, 53]
    lab = ["13", "18", "27", "53", "160"]
    plot_sens(dir_sh, "Number of IoT islands", lab, 0, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n, sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 24, 28, 'true', x_label, 1, 'Number of IoT islands', 50000, False, True)
    x_label = [13, 18, 27, 53]
    lab = ["5", "10", "15", "20"]
    plot_sens(dir_sh, "Number of consumers", lab, 0, sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
              sh_res_dict_icn_icn_n, sh_res_dict_icn_ip_y,
              sh_res_dict_icn_icn_y, 29, 33, 'true', x_label, 1, 'Number of CL1 consumers', 50000, False, True)

    print "Pie charts"
    pie_charts_cache(dir_sh, "SH cache hits", sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
                     sh_res_dict_icn_icn_n, sh_res_dict_icn_ip_y, sh_res_dict_icn_icn_y)
    print "Network usage bars"
    bar_network_usage(dir_sh, "SH network usage", sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
                      sh_res_dict_icn_icn_n,
                      sh_res_dict_icn_ip_y,
                      sh_res_dict_icn_icn_y, 50000)
    print "Cache usage bar"
    bar_cache_usage(dir_sh, "SH cache usage", sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
                    sh_res_dict_icn_icn_n,
                    sh_res_dict_icn_ip_y,
                    sh_res_dict_icn_icn_y)

    ## Baselines
    raw_delay_hops = load_base_line_raw(path_sh, 10, 10000)

    # Delay
    for cntr in range(2):
        box_data.append(raw_delay_hops[0][cntr][1])
    for cntr in range(4):
        box_data.append(raw_delay_hops[1][cntr][1])

    violin_plot_delay(dir_sh, "Delay", box_data[1:], sh_res_dict_ip_x_n, sh_res_dict_ip_x_y, sh_res_dict_icn_ip_n,
                      sh_res_dict_icn_icn_n,
                      sh_res_dict_icn_ip_y, sh_res_dict_icn_icn_y)
    # box_plot(dir_sh, ""box_data[1:])

    # Hops
    box_data = [[]]
    for cntr in range(2):
        box_data.append(raw_delay_hops[0][cntr][0])
    for cntr in range(4):
        box_data.append(raw_delay_hops[1][cntr][0])

    violin_plot_hops(dir_sh, "Hops", box_data[1:])

##
## Smart Factory Parsing
##
print "\nParsing Smart Factory..."
if runs_fa:
    idx = 1

    path = "SF1_ITR4"

    while idx < runs_fa:
        res_container = parse_run(idx, itrs_fa, path, 1)
        fa_res_dict_ip_x_n.append(res_container[0])
        fa_res_dict_ip_x_y.append(res_container[1])
        fa_res_dict_icn_ip_n.append(res_container[2])
        fa_res_dict_icn_icn_n.append(res_container[3])
        fa_res_dict_icn_ip_y.append(res_container[4])
        fa_res_dict_icn_icn_y.append(res_container[5])
        idx += 6
    print "Sensitivity graphs..."
    # Sensitivity graphs
    x_label = [0, 5, 30, 60, 300]
    lab = ["0 s", "5 s", "30 s", "1 min", "5 min"]
    plot_sens(dir_fa, "Data freshness period", lab, 1, fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
              fa_res_dict_icn_icn_n,
              fa_res_dict_icn_ip_y,
              fa_res_dict_icn_icn_y, 1, 6, 'true', x_label, 1, 'Freshness setting (s)', 2000, True, True)
    x_label = [0, 0.5, 0.84, 1]
    lab = ["0", "0.5", "0.84", "1.0"]
    plot_sens(dir_fa, "Content popularity distribution", lab, 0.64, fa_res_dict_ip_x_n, fa_res_dict_ip_x_y,
              fa_res_dict_icn_ip_n,
              fa_res_dict_icn_icn_n, fa_res_dict_icn_ip_y,
              fa_res_dict_icn_icn_y, 6, 10, 'true', x_label, 1, r'Zipf parameter ($\alpha$)', 2000, True, True)
    x_label = [0, 10, 100, 1000, 2000]
    lab = ["0 packets", "10 packets", "100 packets", "1000 packets", "2000 packets"]
    plot_sens(dir_fa, "Cache size", lab, 590, fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
              fa_res_dict_icn_icn_n, fa_res_dict_icn_ip_y,
              fa_res_dict_icn_icn_y, 11, 16, 'true', x_label, 1, 'Cache size (Packets)', 2000, True, True)
    x_label = [0.5, 2, 10, 30, 60]
    lab = ["0.5 s", "2 s", "10 s", "30 s", "60 s"]
    plot_sens(dir_fa, "Request frequency", lab, 0, fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
              fa_res_dict_icn_icn_n, fa_res_dict_icn_ip_y,
              fa_res_dict_icn_icn_y, 16, 21, 'true', x_label, 1, 'Transmission frequency (s)', 2000, False, True)
    x_label = [1, 2, 10, 20, 50]
    lab = ["1\nisland", "2\nislands", "10\nislands", "20\nislands", "50\nislands"]
    plot_sens(dir_fa, "Number of IoT islands", lab, 5, fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
              fa_res_dict_icn_icn_n, fa_res_dict_icn_ip_y,
              fa_res_dict_icn_icn_y, 21, 26, 'true', x_label, 1, '# of islands', 2000, False, True)
    x_label = [50, 100, 150, 200, 250]
    lab = ["50", "100", "150", "200", "250"]
    plot_sens(dir_fa, "Number of consumers", lab, 125, fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
              fa_res_dict_icn_icn_n, fa_res_dict_icn_ip_y,
              fa_res_dict_icn_icn_y, 26, 31, 'true', x_label, 1, 'Number of consumers', 2000, False, True)
    print "Pie chart..."
    pie_charts_cache(dir_fa, "fa cache hits", fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
                     fa_res_dict_icn_icn_n, fa_res_dict_icn_ip_y, fa_res_dict_icn_icn_y)
    print "NWU chart..."
    bar_network_usage(dir_fa, "fa network usage", fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
                      fa_res_dict_icn_icn_n,
                      fa_res_dict_icn_ip_y,
                      fa_res_dict_icn_icn_y, 2000)
    print "CU bar chart..."
    bar_cache_usage(dir_fa, "fa cache usage", fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
                    fa_res_dict_icn_icn_n,
                    fa_res_dict_icn_ip_y,
                    fa_res_dict_icn_icn_y)

    ## Baselines
    print "Loading baselines..."
    raw_delay_hops = load_base_line_raw(path, 5, 400)

    # Delay
    box_data = [[]]
    for cntr in range(2):
        box_data.append(raw_delay_hops[0][cntr][1])
    for cntr in range(4):
        box_data.append(raw_delay_hops[1][cntr][1])
    print "Violin..."
    violin_plot_delay(dir_fa, "Delay", box_data[1:], fa_res_dict_ip_x_n, fa_res_dict_ip_x_y, fa_res_dict_icn_ip_n,
                      fa_res_dict_icn_icn_n,
                      fa_res_dict_icn_ip_y, fa_res_dict_icn_icn_y)

    # Hops
    box_data = [[]]
    for cntr in range(2):
        box_data.append(raw_delay_hops[0][cntr][0])
    for cntr in range(4):
        box_data.append(raw_delay_hops[1][cntr][0])
    print "Hops..."
    violin_plot_hops(dir_fa, "Hops", box_data[1:])

    print "Done!"

    # plt.show()
