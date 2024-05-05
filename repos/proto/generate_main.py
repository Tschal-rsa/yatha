from datetime import datetime, date
import logging
import os

from joblib import Parallel, delayed
from numpy.typing import NDArray
import pandas as pd

from ..brainfeatures.feature_generation.generate_features import generate_features_of_one_file, default_feature_generation_params
from ..brainfeatures.utils.file_util import pandas_store_as_h5
from ..brainfeatures.data_set.edf_abnormal import EDFAbnormal
from ..brainfeatures.utils.sun_grid_engine_util import determime_curr_file_id

root_dir = "/data/lzy/脑电/Processed/"
levels = ["mild_blue/","moderate_orange/","normal_green/","severe_red/",]
splits = ["val","train"]
metric = "cwt"

def process_one_file(data_set, file_id, out_dir, domains, epoch_duration_s,
                     max_abs_val, window_name, band_limits, agg_mode,
                     discrete_wavelet, continuous_wavelet, band_overlap) -> tuple[str, str, NDArray] | None:
    file_name = data_set.file_names[file_id]
    signals, sfreq, pathological = data_set[file_id]
    # print("signals",signals)
    # print("sfreq",sfreq)
    # print("pathological",pathological)
    # exit(0)
    age = data_set.ages[file_id]
    gender = data_set.genders[file_id]

    dirname, file_0 = os.path.split(file_name)
    file_patient_id = file_0.split('.edf_')[0]
    file_number = file_0.split('_')[1].split('.')[0]
    file_label = os.path.basename(dirname)

    feature_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)

    if feature_df is None:
        logging.error("feature generation failed for {}".format(file_id))
        return
    
    return file_patient_id, file_label, feature_df.values

    # also include band limits, epoch_duration_s, etc in additional info?
    additional_info = {
        "sfreq": sfreq,
        "pathological": pathological,
        "age": age,
        "gender": gender,
        "n_samples": signals.shape[1],
        "id": file_id,
        "n_windows": len(feature_df),
        "n_features": len(feature_df.columns),
        "agg": agg_mode,
        "name": file_name,
    }
    info_df = pd.DataFrame(additional_info, index=[0])

    #new_file_name = out_dir+"{:04d}.h5".format(file_id)
    new_file_name = out_dir+file_patient_id+'_'+file_number+".h5"

    pandas_store_as_h5(new_file_name, feature_df, "data")
    pandas_store_as_h5(new_file_name, info_df, "info")


def generate_features_main(data_set, out_dir, domains,
                           run_on_cluster, feat_gen_params, n_jobs):
    today, now = date.today(), datetime.time(datetime.now())
    logging.info('started on {} at {}'.format(today, now))

    # use this to run on cluster. otherwise just give the id of the file that
    # should be cleaned
    if run_on_cluster:
        logging.info("using file id based on sge array job id")
        file_id = determime_curr_file_id(data_set, file_id=None)

        if type(file_id) is not int:
            logging.error(file_id)
            exit()

        process_one_file(data_set, file_id, out_dir, domains,
                         **feat_gen_params)
    else:
        file_ids = range(0, len(data_set))
        Parallel(n_jobs=n_jobs)(
            delayed(process_one_file)
            (data_set, file_id, out_dir, domains, **feat_gen_params) for
            file_id in file_ids)
        # for file_id in file_ids:
        #     process_one_file(data_set, file_id, out_dir, domains, **feat_gen_params)

    today, now = date.today(), datetime.time(datetime.now())
    logging.info('finished on {} at {}'.format(today, now))

if __name__ == "__main__":
    for split in splits:
        for level in levels:
            data_dir = root_dir+"raw/"+split+"/"+level
            output_dir = root_dir+"feature/"+metric+"/"+split+"/"+level

            edf_abnormal = EDFAbnormal(data_dir, ".edf", subset=split)
            edf_abnormal.load()

            default_feature_generation_params["agg_mode"] = "median"  # "median" / "mean"...
            generate_features_main(
                data_set=edf_abnormal,
                out_dir=output_dir,
                domains=[metric],
                run_on_cluster=False,
                feat_gen_params=default_feature_generation_params,
                n_jobs=4
            )
            #["cwt", "dwt", "dft", "phase", "time"]
