import os
import shutil
import pandas as pd
import numpy as np

import cma
import pycst_ctrl
from pycst_data_analyser import PyCstDataAnalyser


class PyCstProjSimulator:
    """ Define the simulation project"""

    def __init__(self, proj_cfg_dic, data_analyser_cfg_dic, optimizer_cfg_dic):

        # Project configuration
        # Settings from users
        self.env = proj_cfg_dic.get('env', 'hpc')
        self.solver = proj_cfg_dic.get('solver', 'T')
        self.gpu_num = proj_cfg_dic.get('gpu_num', 1)
        # Indicate if this is an optimization resumed from previous one
        self.resume_flag = proj_cfg_dic.get('resume_flag', False)
        self.project_folder = proj_cfg_dic.get('project_folder',
                                               "/data/home/eex181/SmoothWallHornProfileRelative_Python")
        self.sim_name = proj_cfg_dic.get('sim_name', "SmoothWallHornProfileRelative_Python")
        self.base_para_file_name = proj_cfg_dic.get('base_para_file_name', "ProfiledSmoothWallHornRelativePara.txt")
        # CST project controlled by matlab will be Farfields
        self.ff_export_subfolder_name = proj_cfg_dic.get('ff_export_subfolder_name', "Farfield")

        # Record csv files of parameters and objective values during optimization
        self.rec_para_file_name = proj_cfg_dic.get('rec_para_file_name', 'opt_para_list.csv')
        self.rec_para_name_lst = proj_cfg_dic.get('rec_para_name_lst', ['EMPTY'])

        self.rec_objval_file_name = proj_cfg_dic.get('rec_objval_file_name', 'opt_obj_val.csv')
        self.rec_obj_name_lst = proj_cfg_dic.get('rec_obj_name_lst', ['EMPTY'])

        # Define cma object saved file name for resuming
        self.cma_object_save_file_name = proj_cfg_dic.get('cma_object_save_file_name', '_saved-cma-object.pkl')
        self.full_cma_object_save_file = os.path.join(self.project_folder, self.cma_object_save_file_name)

        # Initialize run_id
        self.run_id = proj_cfg_dic.get('init_run_id', 1)  # Default run_id starts from 1

        # Settings to generate
        self.cst_file_name = self.sim_name + '.cst'
        self.export_path = os.path.join(self.project_folder, self.sim_name, 'Export')
        self.export_py_path = os.path.join(self.project_folder, self.sim_name, 'ExportPy')
        self.parafile_subfolder_name = 'ParaFiles'
        self.full_sim_para_file = os.path.join(self.project_folder, 'CurrentPara.txt')
        self.full_cst_file = os.path.join(self.project_folder, self.cst_file_name)

        # Create data analyser instance
        self.data_analyser_ins = PyCstDataAnalyser(data_analyser_cfg_dic)

        # Create optimizer instance
        self.optimizer_ins = self.func_opt_cma_create(optimizer_cfg_dic)

    def func_opt_cma_create(self, optimizer_cfg_dic):

        if self.resume_flag is False:
            # Create optimizer object
            cma_init_para_list = optimizer_cfg_dic.get('cma_init_para_list', None)
            cma_sigma = optimizer_cfg_dic.get('cma_sigma', 0.3)
            cma_opts = optimizer_cfg_dic.get('cma_opts', None)
            es = cma.CMAEvolutionStrategy(cma_init_para_list, cma_sigma, cma_opts)
        else:
            # resuming from saved file
            es = pycst_ctrl.func_load_ins_from_file(self.full_cma_object_save_file)

            # overwrite the log directory for outcmaes to current project directory
            # because cmaes recorded the absolute path in es.logger.name_prefix
            es.logger.name_prefix = os.path.join(self.project_folder, 'outcmaes')
            es.logger.name_prefix = es.logger.name_prefix + os.sep

            # Set max iteration number when we know the approximate iteration number for one run (10-day)
            cma_opts = optimizer_cfg_dic.get('cma_opts', None)
            if cma_opts is not None:
                max_iter_num = cma_opts.get('maxiter', None)
                if max_iter_num is not None:
                    es.opts['maxiter'] = max_iter_num

        return es

    @staticmethod
    def func_cst_sim_run(env, solver, gpu_num, full_para_file, full_cst_file):

        if solver == 'F':
            solver_cmd = '-f'
        elif solver == 'T_GPU':
            solver_cmd = '-r -withgpu=%d' % gpu_num
        elif solver == 'I_GPU':
            solver_cmd = '-q -withgpu=%d' % gpu_num
        elif solver == 'T':
            solver_cmd = '-r'
        else:
            solver_cmd = '-r'

        if env == "linux_pc":  # Linux_PC:
            cmd_cst = '"/opt/cst/CST_STUDIO_SUITE_2019/cst_design_environment" -m -par "%s" -r "%s"' % (
                full_para_file, full_cst_file)
        elif env == "hpc":
            cmd_cst = "singularity run --nv /data/containers/cst/cst cst_design_environment -numthreads=${NSLOTS} " \
                      "-m -par \"%s\" %s \"%s\"" % (full_para_file, solver_cmd, full_cst_file)
        elif env == "win":
            cmd_cst = '"C:\\Program Files (x86)\\CST STUDIO SUITE 2019\\CST DESIGN ENVIRONMENT.exe" ' \
                      '-m -par \"%s\" %s \"%s\"' % (full_para_file, solver_cmd, full_cst_file)
        elif env == "test":
            cmd_cst = ""
        else:
            cmd_cst = ""

        if cmd_cst != "":
            if env == "win":
                os.system('"' + cmd_cst + '"')
            else:
                os.system(cmd_cst)

    @staticmethod
    def func_cst_parafile_gen(project_folder, parafile_subfolder_name, base_para_file_name, run_id, cst_para_val_vec):

        # Get full path to the base para file
        full_para_file = os.path.join(project_folder, parafile_subfolder_name, base_para_file_name)

        # Update parameters and generate new para file
        with open(full_para_file, 'r') as fread:
            content_list = fread.readlines()

            for i in range(len(cst_para_val_vec)):
                para_val = cst_para_val_vec[i]
                para_set_str = content_list[i]
                para_set_str = para_set_str % para_val
                content_list[i] = para_set_str

        para_sweep_file_name = os.path.splitext(base_para_file_name)[0] + '_' + str(run_id) + '.txt'
        full_para_sweep_file = os.path.join(project_folder, parafile_subfolder_name, para_sweep_file_name)

        with open(full_para_sweep_file, 'w') as fwrite:
            fwrite.writelines(content_list)

        # Prepare the para file for current simulation
        full_cur_para_file = os.path.join(project_folder, 'CurrentPara.txt')
        shutil.copy(full_para_sweep_file, full_cur_para_file)

    @staticmethod
    def func_cst_opt_res_save(project_folder, para_list_file_name, obj_val_file_name, para_val_vec, para_name_list,
                              obj_val_vec, obj_name_list, run_id):

        # Convert ndarray to pandas series
        para_df = pd.DataFrame(data=para_val_vec.reshape(1, -1), index=[run_id], columns=para_name_list)
        obj_df = pd.DataFrame(data=obj_val_vec.reshape(1, -1), index=[run_id], columns=obj_name_list)

        # prepare the file to write
        full_para_list_file = os.path.join(project_folder, para_list_file_name)
        full_obj_val_file = os.path.join(project_folder, obj_val_file_name)

        if 1 != run_id:
            write_mode = 'a'  # append if already exists
            header_mode = False
        else:
            write_mode = 'w'  # make a new file if not
            header_mode = True

        para_df.to_csv(full_para_list_file, mode=write_mode, header=header_mode)
        obj_df.to_csv(full_obj_val_file, mode=write_mode, header=header_mode)

    def func_cst_proj_sim(self, opt_para_val_vec):

        # Read project configuration
        project_folder = self.project_folder
        export_path = self.export_path
        export_py_path = self.export_py_path
        ff_export_subfolder_name = self.ff_export_subfolder_name
        parafile_subfolder_name = self.parafile_subfolder_name
        base_para_file_name = self.base_para_file_name
        full_sim_para_file = self.full_sim_para_file
        full_cst_file = self.full_cst_file

        # Get current run_id
        run_id = self.run_id

        # Create parameter list
        cst_para_val_vec = opt_para_val_vec

        # Generate the parameter file for simulation
        self.func_cst_parafile_gen(project_folder, parafile_subfolder_name, base_para_file_name, run_id,
                                   cst_para_val_vec)

        # Run the simulation
        self.func_cst_sim_run(self.env, self.solver, self.gpu_num, full_sim_para_file, full_cst_file)

        if self.env == "test":
            # objval_vec = np.random.random_sample((5,)) * 5
            # fitness = objval_vec.sum()
            export_sim_subfolder_name = 'run' + str(run_id)
            bak_sim_subfolder_path = os.path.join(export_py_path, export_sim_subfolder_name)
            # Analyse the export data
            fitness, objval_vec = self.data_analyser_ins.func_cst_data_analyse(bak_sim_subfolder_path, run_id,
                                                                               ff_export_subfolder_name)
        else:
            # Backup the export data of the current simulation
            export_sim_subfolder_name = 'run' + str(run_id)
            bak_sim_subfolder_path = os.path.join(export_py_path, export_sim_subfolder_name)
            shutil.copytree(export_path, bak_sim_subfolder_path)

            # Analyse the export data
            fitness, objval_vec = self.data_analyser_ins.func_cst_data_analyse(bak_sim_subfolder_path, run_id,
                                                                               ff_export_subfolder_name)

        # Save parameter values and objective values of this simulation to .csv record files
        rec_objval_vec = np.append(objval_vec, fitness)
        self.func_cst_opt_res_save(project_folder, self.rec_para_file_name, self.rec_objval_file_name,
                                   opt_para_val_vec, self.rec_para_name_lst,
                                   rec_objval_vec, self.rec_obj_name_lst, run_id)

        # Save cma object to file for resume later
        pycst_ctrl.func_save_ins_to_file(project_folder, self.cma_object_save_file_name, self.optimizer_ins)

        # Update run_id
        self.run_id += 1

        return fitness
