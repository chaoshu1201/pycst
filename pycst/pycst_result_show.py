import os
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse

import pycst_ctrl
from pycst_data_analyser import PyCstDataAnalyser


def func_cst_exp_acii_data_proc(exp_data_dir, exp_file_name):

    # Get full path of the exported data file
    full_exp_data_file = os.path.join(exp_data_dir, exp_file_name)

    # Load data
    headerlin = 0
    exp_data = np.genfromtxt(full_exp_data_file, skip_header=headerlin)

    return exp_data


def func_cst_exp_ff_ar_data_proc(exp_data_folder, singlerun_data_folder_name, ff_ar_filename):

    # Get export S Para data file
    full_ff_ar_file = os.path.join(exp_data_folder, singlerun_data_folder_name, ff_ar_filename)

    # Load data
    headerlin = 0
    ff_ar_data = np.genfromtxt(full_ff_ar_file, skip_header=headerlin)

    # Parses data
    freq_vec = ff_ar_data[:, 0]
    ff_ar_db_vec = ff_ar_data[:, 1]

    return freq_vec, ff_ar_db_vec


def func_spara_group_show(group_data_folder, spara_filename, run_id_str_show_lst):

    plt.figure()
    for run_id_str in run_id_str_show_lst:
        # Get folder name of a single run
        singlerun_data_folder_name = run_id_str

        # Get export S Para data file
        full_spara_file = os.path.join(group_data_folder, singlerun_data_folder_name, spara_filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        s_mag_db = 20 * np.log10(s_mag_lin)

        plt.plot(freq_vec, s_mag_db, linewidth=1, label=singlerun_data_folder_name)

    plt.minorticks_on()
    plt.grid(True, which='both', linestyle=':')
    plt.xlim(85, 115)
    plt.ylim(-60, 0)
    plt.xlabel('Frequency (GHz)')
    plt.ylabel('Magnitude (db)')
    plt.title('S-Parameters')
    plt.legend()
    plt.show()


def func_matlab_style(plot_ax):

    plot_ax.tick_params(axis='both', which='both', direction='in')
    plot_ax.spines['bottom'].set_linewidth(1)
    plot_ax.spines['left'].set_linewidth(1)
    plot_ax.spines['right'].set_linewidth(1)
    plot_ax.spines['top'].set_linewidth(1)
    plot_ax.minorticks_on()
    plot_ax.grid(True, which='both', ls=':', lw=0.5)


def func_group_data_plot(ax, data_plot_cfg_dic_lst, axes_cfg_dic, projection='rectilinear'):
    """
    :param ax: a single axes object
    :param data_plot_cfg_dic_lst:
    data_plot_cfg_dic = {'x_data_vec': ,                            # Only for projection=rectilinear
                         'y_data_group_dic': {'label1': data1,      # Only for projection=rectilinear
                                              'label2': data2},
                         'theta_data_vec': ,                        # Only for projection=polar
                         'r_data_group_dic': {'label1': data1,      # Only for projection=polar
                                              'label2': data2},
                         'color_list': ['r', 'b'],
                         'line_style': ['--', '-'],
                         'marker': ['o', '^'],
                         'marker_edge_color': ['k', 'k'],
                         'markevery': list[int]
                         'line_width': 2
                        }
    :param axes_cfg_dic:
    axes_cfg_dic = {'xlim': [0, 10],    # Only for projection=rectilinear
                    'ylim': [0, 10],    # Only for projection=rectilinear
                    'rlim': [0, 10],    # Only for projection=polar
                    'yticks_vec': np.arange(),  # Only for projection=rectilinear
                    'rticks_vec': np.arange(),  # Only for projection=polar
                    'legend_loc': 'best',
                    'annotation': None
                    }
    :param projection: projection of the current axes, 'rectilinear' or 'polar'. default is 'rectilinear'
    :return:
    """

    for data_plot_cfg_dic in data_plot_cfg_dic_lst:
        if 'polar' == projection:
            x_data_vec = data_plot_cfg_dic['theta_data_vec']
            y_data_group_dic = data_plot_cfg_dic['r_data_group_dic']
            y_data_num = len(y_data_group_dic)
        else:   # 'rectilinear == projection'
            # Get x and y data
            x_data_vec = data_plot_cfg_dic['x_data_vec']
            y_data_group_dic = data_plot_cfg_dic['y_data_group_dic']
            y_data_num = len(y_data_group_dic)

        # Get plot configuration
        line_width = data_plot_cfg_dic.get('line_width', 2)
        color_q = deque(data_plot_cfg_dic.get('color_list', [None]*y_data_num))
        line_style_q = deque(data_plot_cfg_dic.get('line_style', [None]*y_data_num))
        marker_q = deque(data_plot_cfg_dic.get('marker', [None]*y_data_num))
        markevery = data_plot_cfg_dic.get('markevery', None)
        mec_q = deque(data_plot_cfg_dic.get('marker_edge_color', [None]*y_data_num))

        # Plot all the data in data group
        for y_data_label, y_data_vec in y_data_group_dic.items():
            color = color_q.popleft()
            line_style = line_style_q.popleft()
            marker = marker_q.popleft()
            mec = mec_q.popleft()
            ax.plot(x_data_vec, y_data_vec, color=color, ls=line_style, lw=line_width, label=y_data_label,
                    marker=marker, mec=mec, mfc=color, markevery=markevery)

    # Get axes configuration
    xlim_lst = axes_cfg_dic.get('xlim', None)
    ylim_lst = axes_cfg_dic.get('ylim', None)
    rlim_lst = axes_cfg_dic.get('rlim', None)
    legend_loc = axes_cfg_dic.get('legend_loc', 'best')
    annotation = axes_cfg_dic.get('annotation', None)
    yticks_vec = axes_cfg_dic.get('yticks_vec', None)
    rticks_vec = axes_cfg_dic.get('rticks_vec', None)

    # Configure the axes
    if xlim_lst is not None:
        ax.set_xlim(xlim_lst[0], xlim_lst[1])
    if ylim_lst is not None:
        ax.set_ylim(ylim_lst[0], ylim_lst[1])
    if rlim_lst is not None:
        ax.set_rlim(rlim_lst[0], rlim_lst[1])
    if yticks_vec is not None:
        ax.set_yticks(yticks_vec)
    if rticks_vec is not None:
        ax.set_rticks(rticks_vec)

    if 'polar' == projection:
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)      # clockwise

    # Add legends
    ax.legend(loc=legend_loc)

    # Add annotation
    if annotation is not None:
        ax.add_artist(annotation)


def func_group_data_show(data_plot_cfg_dic_lst, axes_cfg_dic, x_label, y_label, fig_title=None, figsize=None):

    # Create a figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot the group data
    func_group_data_plot(ax, data_plot_cfg_dic_lst, axes_cfg_dic)

    # Config the figure
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if fig_title is not None:
        ax.set_title(fig_title, fontweight='bold')

    # Define appearance
    func_matlab_style(ax)
    # ax.set_aspect('equal')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def func_group_data_subplot_show(data_plot_cfg_dic_lst, axes_cfg_dic, nrow, ncol, x_label, y_label, sub_titles,
                                 fig_title=None):

    # Create a figure
    fig, axs = plt.subplots(nrow, ncol)

    # Plot the group data
    for index, ax in enumerate(axs.flat):
        func_group_data_plot(ax, [data_plot_cfg_dic_lst[index]], axes_cfg_dic)

        # Config the figure
        ax.get_legend().remove()    # Remove individual legend for each subplot
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.label_outer()
        if fig_title is not None:
            ax.set_title(fig_title, fontweight='bold')

        # Define appearance
        func_matlab_style(ax)
        ax.text(-25, -43, sub_titles[index], fontsize=11, fontweight='bold')
        # ax.set_aspect('equal')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc='lower center', prop={'size': 8})
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()


def func_cst_radpat_cocx_subplot_show(np_theta, np_phi, pol_ind, ff_data_dir, ff_data_file_dic, nrow, ncol,
                                      axes_cfg_dic):
    """
    This function is used to plot radiation patterns with co-pol and cx-pol
    :param np_theta: number of sample points for theta, normally 360
    :param np_phi: number of sample points for phi, if step is 45deg, np_phi is 5
    :param pol_ind: 'LHCP', 'RHCP' or 'lin_dir'
    :param ff_data_dir: directory path which contains farfield data
    :param ff_data_file_dic: a dictionary list of radiation patterns to be shown, shoud be like:
    ff_data_file_dic = {'75 GHz': "farfield_(f=75)_[1].txt",
                        '85 GHz': "farfield_(f=85)_[1].txt",
                        '90 GHz': "farfield_(f=90)_[1].txt",
                        '95 GHz': "farfield_(f=95)_[1].txt",
                        '100 GHz': "farfield_(f=100)_[1].txt",
                        '110 GHz': "farfield_(f=110)_[1].txt"
                        }
    :param nrow: number of rows
    :param ncol: number of columns
    :param axes_cfg_dic:  see func_group_data_plot() for details
    :return:
    """

    # Create label (legend) list according to the polarization
    if pol_ind == 'LHCP':
        lable_lst = ['LHCP, ' + r'$\phi=0\degree$', 'LHCP, ' + r'$\phi=45\degree$',
                     'LHCP, ' + r'$\phi=90\degree$', 'RHCP, ' + r'$\phi=45\degree$']
    elif pol_ind == 'RHCP':
        lable_lst = ['RHCP, ' + r'$\phi=0\degree$', 'RHCP, ' + r'$\phi=45\degree$',
                     'RHCP, ' + r'$\phi=90\degree$', 'LHCP, ' + r'$\phi=45\degree$']
    else:
        lable_lst = ['Co-pol, ' + r'$\phi=0\degree$', 'Co-pol, ' + r'$\phi=45\degree$',
                     'Co-pol, ' + r'$\phi=90\degree$', 'Cx-pol, ' + r'$\phi=45\degree$']

    data_plot_cfg_dic_lst = []
    for freq_str, ff_file_name in ff_data_file_dic.items():

        # Parse the exported farfield data file
        theta_vec, dir_co_arr, dir_cx_arr, dir_abs_arr, \
            dir_co_norm_arr, dir_cx_norm_arr, dir_abs_norm_arr, \
            peak_co_cvec, peak_cx_cvec, peak_abs_cvec, \
            ar_arr, ar_boresight_cvec = \
            PyCstDataAnalyser.func_exp_ff_data_proc(ff_data_dir, ff_file_name, np_theta, np_phi, pol_ind)

        # Create plot configuration
        data_plot_cfg_sim_radpat_dic = {'x_data_vec': theta_vec,
                                        'y_data_group_dic': {lable_lst[0]: dir_co_norm_arr[0, :],
                                                             lable_lst[1]: dir_co_norm_arr[1, :],
                                                             lable_lst[2]: dir_co_norm_arr[2, :],
                                                             lable_lst[3]: dir_cx_norm_arr[1, :]},
                                        'color_list': ['r', 'k', 'b', 'k'],
                                        'line_style': ['-', '-', '-', '--'],
                                        # 'marker': [None]*4,
                                        # 'marker_edge_color': None,
                                        'line_width': 2
                                        }

        # Create data list, one data_dic for each subplot
        data_plot_cfg_dic_lst.append(data_plot_cfg_sim_radpat_dic)

    x_label = 'Theta (deg)'
    if pol_ind in ['LHCP', 'RHCP']:
        y_label = 'Directivity (dBic)'
    else:
        y_label = 'Directivity (dBi)'
    sub_titles = [sub_title for sub_title in ff_data_file_dic.keys()]

    func_group_data_subplot_show(data_plot_cfg_dic_lst, axes_cfg_dic, nrow, ncol, x_label, y_label, sub_titles)


def func_group_data_subplot_polar_show(data_plot_cfg_dic_lst, axes_cfg_dic, nrow, ncol, sub_titles, fig_title=None):

    # Create a figure
    fig, axs = plt.subplots(nrow, ncol, subplot_kw={'projection': 'polar'},
                            gridspec_kw={'wspace': 0.4, 'hspace': 0.3,
                                         'top': 0.95, 'bottom': 0.05, 'left': 0.05, 'right': 0.95},
                            figsize=(ncol, nrow))

    # Plot the group data
    for index, ax in enumerate(axs.flat):
        func_group_data_plot(ax, [data_plot_cfg_dic_lst[index]], axes_cfg_dic, projection='polar')

        # Config the figure
        ax.get_legend().remove()    # Remove individual legend for each subplot
        ax.tick_params(axis='both', labelsize=8)
        # ax.label_outer()
        if fig_title is not None:
            ax.set_title(fig_title, fontweight='bold')

        # Define appearance
        # func_matlab_style(ax)
        ax.text(np.deg2rad(230), -4.5, sub_titles[index], fontsize=10, fontweight='bold')
        # ax.set_aspect('equal')

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', prop={'size': 8})
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    plt.show()


def func_double_yaxis_data_show(data1_plot_cfg_data_dic_lst, data2_plot_cfg_data_dic_lst,
                                ax1_cfg_dic, ax2_cfg_dic,
                                x_label=None, y1_label=None, y2_label=None, fig_title=None):

    # Create a figure
    fig, ax1 = plt.subplots()

    # Plot data1
    func_group_data_plot(ax1, data1_plot_cfg_data_dic_lst, ax1_cfg_dic)

    # Config ax1
    ax1.set_xlabel(x_label)
    ax1.set_ylabel(y1_label, color='k')
    # ax1.tick_params(axis='y', labelcolor='k')
    func_matlab_style(ax1)

    # Create the 2nd axes
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    # Plot data 2
    func_group_data_plot(ax2, data2_plot_cfg_data_dic_lst, ax2_cfg_dic)

    # Config ax2
    ax2.set_ylabel(y2_label, color='k')  # we already handled the x-label with ax1
    # ax2.tick_params(axis='y', labelcolor='k')

    if fig_title is not None:
        plt.title(fig_title, fontweight='bold')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    # plt.show()

    return fig, ax1, ax2


def func_norm_radiation_pattern_freq_sim_show(ff_data_dir, ff_file_name, np_theta, np_phi, axes_cfg_dic, freq_str,
                                              pol_ind):
    """Display simulated radiation pattern in 0/45/90 deg at one frequency"""

    # Parse the exported farfield data file
    theta_vec, dir_co_arr, dir_cx_arr, dir_abs_arr, \
        dir_co_norm_arr, dir_cx_norm_arr, dir_abs_norm_arr, \
        peak_co_cvec, peak_cx_cvec, peak_abs_cvec, \
        ar_arr, ar_boresight_cvec = \
        PyCstDataAnalyser.func_exp_ff_data_proc(ff_data_dir, ff_file_name, np_theta, np_phi, pol_ind)

    # Create plot configuration
    data_plot_cfg_sim_radpat_dic = {'x_data_vec': theta_vec,
                                    'y_data_group_dic': {r'LHCP, $\phi=0\degree$': dir_co_norm_arr[0, :],
                                                         r'LHCP, $\phi=45\degree$': dir_co_norm_arr[1, :],
                                                         r'LHCP, $\phi=90\degree$': dir_co_norm_arr[2, :],
                                                         r'RHCP, $\phi=45\degree$': dir_cx_norm_arr[1, :]},
                                    'color_list': ['r', 'k', 'b', 'k'],
                                    'line_style': ['-', '-', '-', '--'],
                                    # 'marker': [None]*4,
                                    # 'marker_edge_color': None,
                                    'line_width': 2
                                    }

    data_plot_cfg_dic_lst = [data_plot_cfg_sim_radpat_dic]

    x_label = 'Theta (deg)'
    y_label = 'Directivity (dB)'
    fig_title = pol_ind + ' (%s)' % freq_str

    func_group_data_show(data_plot_cfg_dic_lst, axes_cfg_dic, x_label, y_label, fig_title)


def func_radiation_patter_show(singlerun_data_folder, ff_data_sub_folder_name):
    np_theta = 360
    np_phi = 5

    ff_export_folder = os.path.join(singlerun_data_folder, ff_data_sub_folder_name)

    # Get all the farfield export data file
    farfield_data_file_list = pycst_ctrl.func_file_list_get(ff_export_folder, ext='.txt')

    for export_file in farfield_data_file_list:
        theta_vec, dir_co_arr, dir_cx_arr, dir_abs_arr, \
            dir_co_norm_arr, dir_cx_norm_arr, dir_abs_norm_arr, \
            peak_co_cvec, peak_cx_cvec, peak_abs_cvec, \
            ar_arr, ar_boresight_cvec = \
            PyCstDataAnalyser.func_exp_ff_data_proc(ff_export_folder, export_file, np_theta, np_phi)

        plt.figure()
        plt.plot(theta_vec, dir_co_norm_arr[0, :], 'r-')
        plt.plot(theta_vec, dir_co_norm_arr[1, :], 'b-')
        plt.plot(theta_vec, dir_co_norm_arr[2, :], 'k-')
        plt.plot(theta_vec, dir_cx_norm_arr[1, :], 'b--')
        plt.xlim(-180, 180)
        plt.ylim(-40, 0)
        plt.minorticks_on()
        plt.grid(True, which='both', linestyle=':')
        plt.show()


"""
group_data_folder = "G:\\PyCharmProjectDongPC\\TempData\\SelectedDataChain"
# group_data_folder = "G:/CST_Project/SmoothWallHornProfileRelative_Python_VerifyGroup/FinalResults"
ff_sub_folder_name = 'Farfield'
# spara_file_name = 'S-Parameters_S1(1),1(1).txt'
spara_file_name = 'S-Parameters_S2,1.txt'
# spara_file_name = 'S-Parameters_S1,1.txt'
# run_id_show_lst = ['run2568NewPol1278TriGroove', 'run2568NewPol1278TriGroove020220', 'run2568NewPol1278TriGroove020220Refine']
run_id_show_lst = ['ImMatch_L21', 'ImMatch_Fab', 'ImMatch_20_23']
# run_id_show_lst = ['HornOnly_F-Solver', 'HornOnly_T-Solver', 'Horn_TransSmooth_T-Solver', 'Horn_TransImMatch_F-Solver']
# run_id_show_lst = ['run1888_1flare_refine', 'run2568_1flare_trans']

func_spara_group_show(group_data_folder, spara_file_name, run_id_show_lst)

func_radiation_patter_show(group_data_folder, ff_sub_folder_name)
"""
