import numpy as np
from scipy.signal import find_peaks
import os

import pycst_ctrl


class PyCstDataAnalyser:
    """ Used to analyse data exported by CST"""

    def __init__(self, opts):

        # Initialize attributes
        # Polarization indicator
        self.pol_ind = opts.get('pol_ind', 'lin_dir')

        # Samples in CST farfield data
        self.np_theta = opts.get('np_theta', 360)
        self.np_phi = opts.get('np_phi', 5)

        # Normalization factor for all objective value,
        # all object values will be normalized by (goal_val*norm_factor), which represents the relative tolerance
        # for each goal. This factor aims to bring multiple types of objective values into same range in
        # one cost function.
        self.norm_factor = opts.get('norm_factor', 0.1)

        # Taper definition in dB
        self.taper = opts.get('taper', -12)

        # Weight for rotational symmetry evaluation
        self.rotsym_goal_val = opts.get('rotsym_goal_val', 0)
        self.rotsym_weight = opts.get('rotsym_weight', 0)

        # Goal and Weight for cx-pol level evaluation
        self.cxlevel_goal_val = opts.get('cxlevel_goal_val', -35)
        self.cxlevel_weight = opts.get('cxlevel_weight', 0)

        # Goal and Weight for taper angle evaluation
        self.taperang_goal_range = opts.get('taperang_goal_range', np.array([10, 24]))
        self.taperang_weight = opts.get('taperang_weight', 0)

        # Goal and Weight for SLL evaluation
        self.sll_goal_val = opts.get('sll_goal_val', -30)
        self.sll_weight = opts.get('sll_weight', 0)

        # Goal and weight for Farfield AR evaluation
        self.ar_ff_goal = opts.get('ar_ff_goal', 0)
        # self.ar_ff_max_goal = opts.get('ar_ff_max_goal', 3)
        self.ar_ff_mae_weight = opts.get('ar_ff_mae_weight', 0)
        self.ar_ff_max_weight = opts.get('ar_ff_max_weight', 0)

        # Frequency range within which the S parameters are requested to be evaluated
        self.spara_eva_freq_range_vec = opts.get('spara_eva_freq_range_vec', np.array([85, 110]))

        # Goal and Weight for S11 evaluation in db value
        self.spara_file_name_lst = opts.get('spara_file_name_lst', ['S-Parameters_S1(1),1(1).txt'])
        self.spara_goal_lst = opts.get('spara_goal_lst', [-40])
        self.spara_mae_weight_lst = opts.get('spara_mae_weight_lst', [0])
        self.spara_maxnorm_weight_lst = opts.get('spara_maxnorm_weight_lst', [0])
        # Goal and Weight for S11 evaluation in linear value
        self.spara_lin_goal_lst = opts.get('spara_lin_goal_lst', [0.01])
        self.spara_lin_mae_weight_lst = opts.get('spara_lin_mae_weight_lst', [0])
        self.spara_lin_max_weight_lst = opts.get('spara_lin_max_weight_lst', [0])
        if len(self.spara_goal_lst) == len(self.spara_file_name_lst):
            self.spara_eval_db = True
        else:
            self.spara_eval_db = False
        if len(self.spara_lin_goal_lst) == len(self.spara_file_name_lst):
            self.spara_eval_lin = True
        else:
            self.spara_eval_lin = False

        # Frequency range within which the NR AR are requested to be evaluated
        self.nf_ar_eva_freq_range_vec = opts.get('nf_ar_eva_freq_range_vec', np.array([85, 110]))

        # Goal and Weight for nearfield AR evaluation
        self.nf_ar_file_name = opts.get('nf_ar_file_name', 'AR_AllFreq.txt')
        self.nf_ar_goal = opts.get('nf_ar_goal', 0)
        self.nf_ar_mae_weight = opts.get('nf_ar_mae_weight', 0)
        self.nf_ar_maxnorm_weight = opts.get('nf_ar_maxnorm_weight', 0)

    @staticmethod
    def func_taper_angle_get(theta_vec, dir_co_norm_arr, taper):
        index_barr = dir_co_norm_arr >= taper
        theta_arr = np.vstack((theta_vec, theta_vec, theta_vec))
        theta_arr_tapered = theta_arr[index_barr]
        taper_ang = np.amax(np.absolute(theta_arr_tapered))

        return taper_ang

    @staticmethod
    def func_exp_ff_data_proc(export_folder, filename, np_theta, np_phi, pol_ind='RHCP'):

        # Get export farfield data file
        full_exp_ff_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 2
        cut_data = np.genfromtxt(full_exp_ff_file, skip_header=headerlin)

        # Get Theta (Col.1)
        theta_arr = cut_data[:, 0].reshape(np_phi, np_theta)
        theta_arr = np.c_[theta_arr, 360 + theta_arr[:, 0]]
        theta_vec = theta_arr[0, :]

        # Decide data column index for co-pol and cx-pol
        if pol_ind == 'LHCP':
            co_col_index = 3
            cx_col_index = 5
        else:   # pol_ind = 'RHCP' or 'lin_dir'
            co_col_index = 5
            cx_col_index = 3

        # Get directivity for Co-Pol (Col.4 is LHCP, Col.6 is RHCP or Co-pol if linear direction is chose) for each cut
        dir_co_all = cut_data[:, co_col_index].reshape(np_phi, np_theta)
        dir_co_arr = dir_co_all[2:5, :]
        dir_co_arr = np.c_[dir_co_arr, dir_co_arr[:, 0]]
        peak_co_cvec = dir_co_arr[:, np_theta // 2, np.newaxis]  # Transform to (3,1) column vector
        dir_co_norm_arr = dir_co_arr - peak_co_cvec

        # Get directivity for Cx-Pol (Col.4 is LHCP, Col.6 is RHCP or Cx-pol if linear direction is chose) for each cut
        dir_cx_all = cut_data[:, cx_col_index].reshape(np_phi, np_theta)
        dir_cx_arr = dir_cx_all[2:5, :]
        dir_cx_arr = np.c_[dir_cx_arr, dir_cx_arr[:, 0]]
        peak_cx_cvec = dir_cx_arr.max(axis=1).reshape(-1, 1)
        dir_cx_norm_arr = dir_cx_arr - peak_co_cvec

        # Get directivity for Abs (Col.3) for each cut
        dir_abs_all = cut_data[:, 2].reshape(np_phi, np_theta)
        dir_abs_arr = dir_abs_all[2:5, :]
        dir_abs_arr = np.c_[dir_abs_arr, dir_abs_arr[:, 0]]
        peak_abs_cvec = dir_abs_arr[:, np_theta // 2, np.newaxis]  # Transform to (3,1) column vector
        dir_abs_norm_arr = dir_abs_arr - peak_abs_cvec

        # Get AR (Col.8) for each cut
        ar_all = cut_data[:, 7].reshape(np_phi, np_theta)
        ar_arr = ar_all[2:5, :]
        ar_arr = np.c_[ar_arr, ar_arr[:, 0]]
        ar_boresight_cvec = ar_arr[:, np_theta // 2, np.newaxis]  # Transform to (3,1) column vector

        return theta_vec, dir_co_arr, dir_cx_arr, dir_abs_arr, \
            dir_co_norm_arr, dir_cx_norm_arr, dir_abs_norm_arr, \
            peak_co_cvec, peak_cx_cvec, peak_abs_cvec, \
            ar_arr, ar_boresight_cvec

    def func_rotsym_objval_calc_mse(self, theta_vec, dir_co_norm_arr, taper, goal_val, weight):
        """
        This function calculates the weighted MSE between radiation pattern of each cut and average radiation pattern,
        the result is a scalar value which represents the rotational symmetry at this frequency.
        The goal and weight have already been considered in the return value at this frequency sample, so the return
        value could be used to calculate truncated MAE over all frequency samples.
        """

        # Get the taper angle
        taper_ang = self.func_taper_angle_get(theta_vec, dir_co_norm_arr, taper)
        angle_range = np.array([-taper_ang, taper_ang])

        # Get theta vector within the taper angle
        index_bvec = np.logical_and(theta_vec >= angle_range[0], theta_vec <= angle_range[1])
        theta_tapered_vec = theta_vec[index_bvec]

        # Calculate weight for directivity at different theta
        radpat_weight_vec = 10 ** ((-1) * (np.absolute(theta_tapered_vec) / taper_ang))

        # Calculate MSE between radiation pattern of each cut and average radiation pattern
        index_barr = np.vstack((index_bvec, index_bvec, index_bvec))
        dir_co_norm_tapered_arr = dir_co_norm_arr[index_barr].reshape(-1, len(theta_tapered_vec))
        dir_co_norm_avg_tapered_vec = np.mean(dir_co_norm_tapered_arr, axis=0)
        # Calculate difference based on array broadcasting
        dir_co_sqrdiff_tapered_arr = (dir_co_norm_tapered_arr - dir_co_norm_avg_tapered_vec) ** 2
        mse_vec = np.mean(dir_co_sqrdiff_tapered_arr * radpat_weight_vec, axis=1)

        # Calculate objective value
        objective_val = max(mse_vec.sum() - goal_val, 0)
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_cxlevel_objval_calc_trunc(dir_cx_norm_arr, goal_val, weight, norm_factor):
        """
        This function calculates the cx-level which is the maximum level of normalized cx-pol among all cuts at one
        frequency.
        The goal and weight have already been considered in the return value at this frequency sample, so the return
        value could be used to calculate truncated MAE over all frequency samples.
        """

        # Get the maximum level of normalized cx-pol among all cuts
        cxlevel = np.amax(dir_cx_norm_arr)

        # Calculate truncated objective value
        objective_val = max((cxlevel - goal_val), 0) / (abs(goal_val) * norm_factor)
        objective_val *= weight

        return objective_val

    def func_taperang_objval_calc_rangetrunc(self, theta_vec, dir_co_norm_arr, taper, goal_range, weight, norm_factor):
        """
        This function calculates the truncated difference between simulated taper angle and expected taper angle range
        at one frequency.
        The goal and weight have already been considered in the return value at this frequency sample, so the return
        value could be used to calculate truncated MAE over all frequency samples.
        """

        # Get the max taper angle of the radiation pattern
        taper_ang = self.func_taper_angle_get(theta_vec, dir_co_norm_arr, taper)

        range_cent = goal_range.mean()
        if (taper_ang >= goal_range[0]) and (taper_ang <= goal_range[1]):
            objective_val = 0   # Objective is 0 if simulated taper angle is in the expected range
        else:
            objective_val = abs(taper_ang - range_cent) / (range_cent * norm_factor)

        objective_val *= weight

        return objective_val

    @staticmethod
    def func_sll_max_get(dir_co_norm_arr):
        """
        This function gets the max SLL of all cuts
        """

        cut_num = dir_co_norm_arr.shape[0]

        # Initialize an array to store SLL value of each cut
        sll_val_vec = np.zeros(cut_num)

        for i in range(cut_num):
            # Get peaks of each cut
            dir_co_norm_cut_vec = dir_co_norm_arr[i, :]
            peak_index, properties = find_peaks(dir_co_norm_cut_vec)
            peaks = dir_co_norm_cut_vec[peak_index]

            # Sort peaks in ascending order
            pks_sort = np.sort(peaks)

            if 0 == len(pks_sort):  # Probably there is no radiation at all due to large reflection coefficient
                sll_val_vec[i] = 65535
            elif 1 == len(pks_sort):
                sll_val_vec[i] = -128  # Set SLL value to a value that will always be lower than the goal
            else:
                sll_val_vec[i] = pks_sort[-2]  # Not always the 1st sidelobe but the highest one

        sll_val_max = np.amax(sll_val_vec)

        return sll_val_max

    def func_sll_objval_calc_trunc(self, dir_co_norm_arr, goal_val, weight, norm_factor):
        """
        This function calculates truncated difference between max SLL of all cuts and expected SLL at this frequency
        The goal and weight have already been considered in the return value at this frequency sample, so the return
        value could be used to calculate truncated MAE over all frequency samples.
        """
        # Get max SLL of all cuts
        sll_val_max = self.func_sll_max_get(dir_co_norm_arr)

        objective_val = max((sll_val_max - goal_val), 0) / (abs(goal_val) * norm_factor)
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_ar_ff_objval_calc_maetrunc(theta_vec, ar_arr, angle_range, goal_val, weight, norm_factor):
        """
        This function calculates the truncated MAE of AR over given beamwidth (angle range) at one frequency
        The goal and weight have already been considered in the return value at this frequency sample, so the return
        value could be used to calculate truncated MAE over all frequency samples.
        """

        # Get AR array within the taper angle
        index_bvec = np.logical_and(theta_vec >= angle_range[0], theta_vec <= angle_range[1])
        theta_tapered_vec = theta_vec[index_bvec]
        index_barr = np.vstack((index_bvec, index_bvec, index_bvec))
        ar_tapered_arr = ar_arr[index_barr].reshape(-1, len(theta_tapered_vec))

        # Get the max AR over the given beamwidth among all cuts
        max_ar_tapered_vec = np.amax(ar_tapered_arr, axis=0)

        # Calculate the truncated difference simulated AR and expected AR over the given beamwidth
        diff_trunc_vec = np.maximum((max_ar_tapered_vec - goal_val), 0)

        # objective_val = diff_trunc_vec.mean() / (abs(goal_val) * norm_factor)
        objective_val = diff_trunc_vec.mean() / (abs(goal_val))
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_ar_ff_objval_calc_maxtrunc(theta_vec, ar_arr, angle_range, goal_val, weight, norm_factor):
        """
        This function calculates truncated difference between max AR over given beamwidth for all cuts at one frequency
        The goal and weight have already been considered in the return value at this frequency sample, so the return
        value could be used to calculate truncated MAE over all frequency samples.
        """
        # Get AR array within the taper angle
        index_bvec = np.logical_and(theta_vec >= angle_range[0], theta_vec <= angle_range[1])
        theta_tapered_vec = theta_vec[index_bvec]
        index_barr = np.vstack((index_bvec, index_bvec, index_bvec))
        ar_tapered_arr = ar_arr[index_barr].reshape(-1, len(theta_tapered_vec))

        # Get the max AR among all cuts
        max_ar_tapered_vec = np.amax(ar_tapered_arr, axis=0)

        # Limit the evaluate frequency range
        diff_trunc_vec = np.maximum((max_ar_tapered_vec - goal_val), 0)

        max_diff_trunc = diff_trunc_vec.max()
        # objective_val = max_diff_trunc / (abs(goal_val) * norm_factor)
        objective_val = max_diff_trunc / (abs(goal_val))
        # objective_val = max_diff_trunc
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_cst_spara_data_proc(exp_data_folder, singlerun_data_folder_name, spara_filename):

        # Get export S Para data file
        full_spara_file = os.path.join(exp_data_folder, singlerun_data_folder_name, spara_filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        s_mag_db = 20 * np.log10(s_mag_lin)

        return freq_vec, s_mag_lin, s_mag_db

    @staticmethod
    def func_spara_objval_calc_maetrunc(export_folder, filename, goal_val, freq_range_vec, weight, norm_factor):

        # Get export S Para data file
        full_spara_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        s_mag_db = 20 * np.log10(s_mag_lin)
        # s_phase = spara_data[:, 2]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_range_vec[0], freq_vec <= freq_range_vec[1])
        diff_trunc_vec = np.maximum((s_mag_db[index_bvec] - goal_val), 0)

        objective_val = diff_trunc_vec.mean() / (abs(goal_val) * norm_factor)
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_spara_objval_calc_maxnormtrunc(export_folder, filename, goal_val, freq_range_vec, weight, norm_factor):

        # Get export S Para data file
        full_spara_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        s_mag_db = 20 * np.log10(s_mag_lin)
        # s_phase = spara_data[:, 2]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_range_vec[0], freq_vec <= freq_range_vec[1])
        diff_trunc_vec = np.maximum((s_mag_db[index_bvec] - goal_val), 0)

        max_diff_trunc = diff_trunc_vec.max()
        objective_val = max_diff_trunc / (abs(goal_val) * norm_factor)
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_spara_lin_objval_calc_maetrunc(export_folder, filename, goal_val, freq_range_vec, weight):

        # Get export S Para data file
        full_spara_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        # s_mag_db = 20 * np.log10(s_mag_lin)
        # s_phase = spara_data[:, 2]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_range_vec[0], freq_vec <= freq_range_vec[1])
        diff_trunc_vec = np.maximum((s_mag_lin[index_bvec] - goal_val), 0)

        objective_val = diff_trunc_vec.mean()
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_spara_lin_objval_calc_maxtrunc(export_folder, filename, goal_val, freq_range_vec, weight):

        # Get export S Para data file
        full_spara_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        # s_mag_db = 20 * np.log10(s_mag_lin)
        # s_phase = spara_data[:, 2]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_range_vec[0], freq_vec <= freq_range_vec[1])
        diff_trunc_vec = np.maximum((s_mag_lin[index_bvec] - goal_val), 0)

        max_diff_trunc = diff_trunc_vec.max()
        objective_val = max_diff_trunc
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_nf_ar_objval_calc_maetrunc(export_folder, filename, goal_val, freq_range_vec, weight):

        # Get export S Para data file
        full_nr_ar_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        nf_ar_data = np.genfromtxt(full_nr_ar_file, skip_header=headerlin)

        # Parses data
        freq_vec = nf_ar_data[:, 0]
        nf_ar_real = nf_ar_data[:, 1]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_range_vec[0], freq_vec <= freq_range_vec[1])
        diff_trunc_vec = np.maximum((nf_ar_real[index_bvec] - goal_val), 0)

        objective_val = diff_trunc_vec.mean()
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_nf_ar_objval_calc_maxnormtrunc(export_folder, filename, goal_val, freq_range_vec, weight):

        # Get export S Para data file
        full_nr_ar_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        nf_ar_data = np.genfromtxt(full_nr_ar_file, skip_header=headerlin)

        # Parses data
        freq_vec = nf_ar_data[:, 0]
        nf_ar_real = nf_ar_data[:, 1]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_range_vec[0], freq_vec <= freq_range_vec[1])
        diff_trunc_vec = np.maximum((nf_ar_real[index_bvec] - goal_val), 0)

        max_diff_trunc = diff_trunc_vec.max()
        objective_val = max_diff_trunc / (abs(goal_val) / 2)
        objective_val *= weight

        return objective_val

    @staticmethod
    def func_spara_data_sample_extract(export_folder, filename, freq_limit_vec, sample_num):

        # Get export S Para data file
        full_spara_file = os.path.join(export_folder, filename)

        # Load data
        headerlin = 0
        spara_data = np.genfromtxt(full_spara_file, skip_header=headerlin)

        # Parses data
        freq_vec = spara_data[:, 0]
        s_mag_lin = spara_data[:, 1]
        # s_mag_db = 20 * np.log10(s_mag_lin)
        # s_phase = spara_data[:, 2]

        # Limit the evaluate frequency range
        index_bvec = np.logical_and(freq_vec >= freq_limit_vec[0], freq_vec <= freq_limit_vec[1])
        freq_range_vec = freq_vec[index_bvec]
        s_mag_range_lin_vec = s_mag_lin[index_bvec]

        # Get the samples from the data within the range
        data_num = len(s_mag_range_lin_vec)  # data points including the first and last points
        # Generate sample index between the first index (0) and the last index (data_num -1) (include last index)
        sample_index = np.linspace(0, data_num - 1, sample_num).astype(int)  # Cast the sample index to integer
        freq_sample_vec = freq_range_vec[sample_index]
        s_mag_lin_sample_vec = s_mag_range_lin_vec[sample_index]

        return freq_sample_vec, s_mag_lin_sample_vec

    def func_cst_data_spara_analyse(self, singlerun_export_folder):

        # Evaluate S parameters
        spara_objval_vec = np.array([])
        spara_eva_freq_range_vec = self.spara_eva_freq_range_vec

        for i in range(len(self.spara_file_name_lst)):
            spara_file_name = self.spara_file_name_lst[i]

            if self.spara_eval_db is True:  # if configuration is valid
                # Goal and weight for spara in db values
                spara_goal_val = self.spara_goal_lst[i]
                spara_mae_weight = self.spara_mae_weight_lst[i]
                spara_maxnorm_weight = self.spara_maxnorm_weight_lst[i]

                if spara_mae_weight != 0:
                    spara_objval_mae = self.func_spara_objval_calc_maetrunc(singlerun_export_folder, spara_file_name,
                                                                            spara_goal_val, spara_eva_freq_range_vec,
                                                                            spara_mae_weight, self.norm_factor)
                    # Form s-para objective value list
                    spara_objval_vec = np.append(spara_objval_vec, spara_objval_mae)

                if spara_maxnorm_weight != 0:
                    spara_objval_max = self.func_spara_objval_calc_maxnormtrunc(singlerun_export_folder,
                                                                                spara_file_name, spara_goal_val,
                                                                                spara_eva_freq_range_vec,
                                                                                spara_maxnorm_weight, self.norm_factor)
                    # Form s-para objective value list
                    spara_objval_vec = np.append(spara_objval_vec, spara_objval_max)

            if self.spara_eval_lin is True:
                # Goal and weight for spara in linear values
                spara_lin_goal_val = self.spara_lin_goal_lst[i]
                spara_lin_mae_weight = self.spara_lin_mae_weight_lst[i]
                spara_lin_max_weight = self.spara_lin_max_weight_lst[i]

                if spara_lin_mae_weight != 0:
                    spara_lin_objval_mae = self.func_spara_lin_objval_calc_maetrunc(singlerun_export_folder,
                                                                                    spara_file_name, spara_lin_goal_val,
                                                                                    spara_eva_freq_range_vec,
                                                                                    spara_lin_mae_weight)
                    # Form s-para objective value list
                    spara_objval_vec = np.append(spara_objval_vec, spara_lin_objval_mae)

                if spara_lin_max_weight != 0:
                    spara_lin_objval_max = self.func_spara_lin_objval_calc_maxtrunc(singlerun_export_folder,
                                                                                    spara_file_name, spara_lin_goal_val,
                                                                                    spara_eva_freq_range_vec,
                                                                                    spara_lin_max_weight)
                    # Form s-para objective value list
                    spara_objval_vec = np.append(spara_objval_vec, spara_lin_objval_max)

        return spara_objval_vec

    def func_cst_data_farfield_analyse(self, singlerun_export_folder, ff_export_sub_folder):

        # Get farfield data file list if it exists
        ff_export_folder = os.path.join(singlerun_export_folder, ff_export_sub_folder)
        if ff_export_sub_folder != "":
            # Get all the farfield export data file
            farfield_data_file_list = pycst_ctrl.func_file_list_get(ff_export_folder, ext='.txt')
        else:
            farfield_data_file_list = ""

        # Initialize result vectors over all frequency samples
        rotsym_objval_vec = np.array([])
        cxlevel_objval_vec = np.array([])
        taperang_objval_vec = np.array([])
        sll_objval_vec = np.array([])
        ar_ff_mae_objval_vec = np.array([])
        ar_ff_max_objval_vec = np.array([])
        for export_file in farfield_data_file_list:
            theta_vec, dir_co_arr, dir_cx_arr, dir_abs_arr, \
            dir_co_norm_arr, dir_cx_norm_arr, dir_abs_norm_arr, \
            peak_co_cvec, peak_cx_cvec, peak_abs_cvec, \
            ar_arr, ar_boresight_cvec = \
                self.func_exp_ff_data_proc(ff_export_folder, export_file, self.np_theta, self.np_phi, self.pol_ind)

            # Calculate rotational symmetry fitness
            if self.rotsym_weight != 0:
                rotsym_objval_freq = self.func_rotsym_objval_calc_mse(theta_vec, dir_co_norm_arr, self.taper,
                                                                      self.rotsym_goal_val, self.rotsym_weight)
                rotsym_objval_vec = np.append(rotsym_objval_vec, rotsym_objval_freq)

            # Calculate Cx-Pol level fitness
            if self.cxlevel_weight != 0:
                cxlevel_objval_freq = self.func_cxlevel_objval_calc_trunc(dir_cx_norm_arr, self.cxlevel_goal_val,
                                                                          self.cxlevel_weight, self.norm_factor)
                cxlevel_objval_vec = np.append(cxlevel_objval_vec, cxlevel_objval_freq)

            # Calculate taper angle fitness
            if self.taperang_weight != 0:
                taperang_objval_freq = self.func_taperang_objval_calc_rangetrunc(theta_vec, dir_co_norm_arr,
                                                                                 self.taper,
                                                                                 self.taperang_goal_range,
                                                                                 self.taperang_weight, self.norm_factor)
                taperang_objval_vec = np.append(taperang_objval_vec, taperang_objval_freq)

            # Calculate SLL fitness
            if self.sll_weight != 0:
                sll_objval_freq = self.func_sll_objval_calc_trunc(dir_co_norm_arr, self.sll_goal_val,
                                                                  self.sll_weight, self.norm_factor)
                sll_objval_vec = np.append(sll_objval_vec, sll_objval_freq)

            # Calculate Farfield AR fitness
            # Decide theta range for evaluation
            taper_ang = self.func_taper_angle_get(theta_vec, dir_co_norm_arr, self.taper)
            angle_range = np.array([-taper_ang, taper_ang])
            # Calculate fitness over the theta range
            if self.ar_ff_mae_weight != 0:
                af_ff_mae_objval_freq = self.func_ar_ff_objval_calc_maetrunc(theta_vec, ar_arr, angle_range,
                                                                             self.ar_ff_goal, self.ar_ff_mae_weight,
                                                                             self.norm_factor)
                ar_ff_mae_objval_vec = np.append(ar_ff_mae_objval_vec, af_ff_mae_objval_freq)
            if self.ar_ff_max_weight != 0:
                af_ff_max_objval_freq = self.func_ar_ff_objval_calc_maxtrunc(theta_vec, ar_arr, angle_range,
                                                                             self.ar_ff_goal, self.ar_ff_max_weight,
                                                                             self.norm_factor)
                ar_ff_max_objval_vec = np.append(ar_ff_max_objval_vec, af_ff_max_objval_freq)

        # Form radiation pattern objective value list
        radpat_objval_vec = np.array([])
        if self.rotsym_weight != 0:
            radpat_objval_vec = np.append(radpat_objval_vec, rotsym_objval_vec.mean())
        if self.cxlevel_weight != 0:
            radpat_objval_vec = np.append(radpat_objval_vec, cxlevel_objval_vec.mean())
        if self.taperang_weight != 0:
            radpat_objval_vec = np.append(radpat_objval_vec, taperang_objval_vec.mean())
        if self.sll_weight != 0:
            radpat_objval_vec = np.append(radpat_objval_vec, sll_objval_vec.mean())
        if self.ar_ff_mae_weight != 0:
            radpat_objval_vec = np.append(radpat_objval_vec, ar_ff_mae_objval_vec.mean())
        if self.ar_ff_max_weight != 0:
            radpat_objval_vec = np.append(radpat_objval_vec, ar_ff_max_objval_vec.mean())

        return radpat_objval_vec

    def func_cst_data_analyse(self, singlerun_export_folder, run_id, ff_export_sub_folder):

        # Evaluate S parameters
        spara_objval_vec = self.func_cst_data_spara_analyse(singlerun_export_folder)

        # Evaluate farfield
        radpat_objval_vec = self.func_cst_data_farfield_analyse(singlerun_export_folder, ff_export_sub_folder)

        # Evaluate near-field AR
        nf_ar_objval_vec = np.array([])
        nf_ar_file_name = self.nf_ar_file_name
        nf_ar_goal_val = self.nf_ar_goal
        nf_ar_mae_weight = self.nf_ar_mae_weight
        nf_ar_maxnorm_weight = self.nf_ar_maxnorm_weight
        nf_ar_eva_freq_range_vec = self.nf_ar_eva_freq_range_vec

        if nf_ar_mae_weight != 0:
            nf_ar_objval_mae = self.func_nf_ar_objval_calc_maetrunc(singlerun_export_folder, nf_ar_file_name,
                                                                    nf_ar_goal_val, nf_ar_eva_freq_range_vec,
                                                                    nf_ar_mae_weight)
            # Form NF AR objective value list
            nf_ar_objval_vec = np.append(nf_ar_objval_vec, nf_ar_objval_mae)

        if nf_ar_maxnorm_weight != 0:
            nf_ar_objval_max = self.func_nf_ar_objval_calc_maxnormtrunc(singlerun_export_folder, nf_ar_file_name,
                                                                        nf_ar_goal_val, nf_ar_eva_freq_range_vec,
                                                                        nf_ar_maxnorm_weight)

            # Form near-field AR objective value list
            nf_ar_objval_vec = np.append(nf_ar_objval_vec, nf_ar_objval_max)

        # Combine all objective values
        objval_vec = np.concatenate((spara_objval_vec, radpat_objval_vec, nf_ar_objval_vec))
        objval_total = objval_vec.sum()

        # print objective values
        objval_vec_str = np.array2string(objval_vec, precision=7, separator=',', suppress_small=True)
        pring_msg = "Sim[%d]: ObjValVec = %s; ObjVal = %f;" % (run_id, objval_vec_str, objval_total)
        print(pring_msg)

        return objval_total, objval_vec
