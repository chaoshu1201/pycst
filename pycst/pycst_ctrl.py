import os
import pickle
import pandas as pd


def func_save_ins_to_file(project_folder, save_file_name, save_obj):
    full_save_file = os.path.join(project_folder, save_file_name)

    with open(full_save_file, 'wb') as fdump:
        pickle.dump(save_obj, fdump)


def func_load_ins_from_file(full_save_file):
    with open(full_save_file, 'rb') as fload:
        es = pickle.load(fload)

    return es


def func_subdir_list_get(parent_dir_name):
    return list(filter(os.path.isdir,
                       map(lambda filename: os.path.join(parent_dir_name, filename),
                           os.listdir(parent_dir_name))))


def func_file_list_get(dirname, ext='.txt'):
    return list(filter(
        lambda filename: os.path.splitext(filename)[1] == ext,
        os.listdir(dirname)))


def func_get_runid_from_dirname(sim_run_dirname):
    # Find start index of run_id after "run" from the dirname "runxxx"
    start_index = sim_run_dirname.find("run") + 3
    run_id = int(sim_run_dirname[start_index:])

    return run_id


def func_cst_opt_res_para_save(project_folder, para_list_file_name, para_val_vec, para_name_list, run_id):

    # Convert ndarray to pandas series
    para_df = pd.DataFrame(data=para_val_vec.reshape(1, -1), index=[run_id], columns=para_name_list)

    # prepare the file to write
    full_para_list_file = os.path.join(project_folder, para_list_file_name)

    if 1 != run_id:
        write_mode = 'a'  # append if already exists
        header_mode = False
    else:
        write_mode = 'w'  # make a new file if not
        header_mode = True

    para_df.to_csv(full_para_list_file, mode=write_mode, header=header_mode)


def func_cst_opt_res_obj_save(project_folder, obj_val_file_name, obj_val_vec, obj_name_list, run_id):

    # Convert ndarray to pandas series
    obj_df = pd.DataFrame(data=obj_val_vec.reshape(1, -1), index=[run_id], columns=obj_name_list)

    # prepare the file to write
    full_obj_val_file = os.path.join(project_folder, obj_val_file_name)

    if 1 != run_id:
        write_mode = 'a'  # append if already exists
        header_mode = False
    else:
        write_mode = 'w'  # make a new file if not
        header_mode = True

    obj_df.to_csv(full_obj_val_file, mode=write_mode, header=header_mode)
