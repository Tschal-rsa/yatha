from datetime import datetime, date
import logging

from joblib import Parallel, delayed
import pandas as pd

from repos.brainfeatures.feature_generation.generate_features import (
    generate_features_of_one_file, default_feature_generation_params)
from repos.brainfeatures.utils.file_util import pandas_store_as_h5
from repos.brainfeatures.data_set.tuh_abnormal import TuhAbnormal
from repos.brainfeatures.data_set.edf_abnormal import EDFAbnormal
from repos.brainfeatures.utils.sun_grid_engine_util import (
    determime_curr_file_id)


def process_one_file(data_set, file_id, out_dir, domains, epoch_duration_s,
                     max_abs_val, window_name, band_limits, agg_mode,
                     discrete_wavelet, continuous_wavelet, band_overlap):
    file_name = data_set.file_names[file_id]
    signals, sfreq, pathological = data_set[file_id]
    age = data_set.ages[file_id]
    gender = data_set.genders[file_id]

    file_0 = file_name.split("/home/zhangrui/Desktop/EEG_Process/Data_2.0/NOTCH3/")[1]
    file_patient_id = file_0.split('.edf_')[0]
    file_number = file_0.split('_')[1].split('.')[0]

    feature_df = generate_features_of_one_file(
        signals, sfreq, epoch_duration_s, max_abs_val, window_name,
        band_limits, agg_mode, discrete_wavelet,
        continuous_wavelet, band_overlap, domains)

    if feature_df is None:
        logging.error("feature generation failed for {}".format(file_id))
        return

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
    log = logging.getLogger()
    log.setLevel("INFO")
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

    today, now = date.today(), datetime.time(datetime.now())
    logging.info('finished on {} at {}'.format(today, now))

if __name__ == "__main__":
    data_dir = "/home/zhangrui/Desktop/EEG_Process/Data_2.0/NOTCH3/"
    output_dir = "/home/zhangrui/Desktop/EEG_Process/Feature/Data_2.0/NOTCH3/"

    edf_abnormal = EDFAbnormal(data_dir, ".edf", subset="NOTCH3")
    edf_abnormal.load()

    default_feature_generation_params["agg_mode"] = None  # "median" / "mean"...
    generate_features_main(
        data_set=edf_abnormal,
        out_dir=output_dir,
        domains=["cwt", "dwt", "dft"],
        run_on_cluster=False,
        feat_gen_params=default_feature_generation_params,
        n_jobs=4
    )
    #["cwt", "dwt", "dft", "phase", "time"]
