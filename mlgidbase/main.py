import numpy as np
from typing import List, Any, Optional
import logging

import pygid
from .pygid_functions import check_valid_conversion, save_detect, get_entry_dict, \
    read_conversion_from_nexus, save_fit, save_match, read_detected_peaks, save_pipeline
from .mlgiddetect_functions import load_config, run_mlgiddetect, load_inference
from .pygidfit_functions import  run_pygidfit_from_file, run_pygidfit_from_memory
from .mlgidmatch_functions import load_cif_prepr, run_mlgidmatch_from_file, solution2container, set_match_class, \
    get_unique_solutions
from mlgidmatch.preprocess.cif_preprocess import CifPattern
from mlgiddetect.configuration import Config
from mlgiddetect.inference import Inference

class mlgidBASE:
    def __init__(
        self,
        *,
        filename: Optional[str] = None,
        pygid_conversion: Optional[pygid.Conversion] = None,
        imp_detect: Optional[Inference] = None,
        config_detect: Optional[Config] = None,
        cif_prepr: Optional[CifPattern] = None,
        path_to_save: str = "result.h5",
        overwrite_file: bool = True,
        h5_group: str = "entry_0000",
        overwrite_group: bool = False,
        smpl_metadata: Any = None,
        exp_metadata: Any = None,
    ):
        self.filename = filename
        self.pygid_conversion = pygid_conversion

        self.imp_detect = imp_detect
        self.config_detect = config_detect

        self.cif_prepr = cif_prepr

        self.path_to_save = path_to_save
        self.overwrite_file = overwrite_file
        self.h5_group = h5_group
        self.overwrite_group = overwrite_group
        self.smpl_metadata = smpl_metadata
        self.exp_metadata = exp_metadata

        self.from_nexus: Optional[bool] = None
        self.entry_dict: Any = None

        self.img_container_detect_list = None
        self.img_container_fit_list = None
        self.container_match_list = None

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.INFO)

        self._validate_input()
        self._initialize_source()

    def _validate_input(self) -> None:
        if self.pygid_conversion is None and self.filename is None:
            raise AttributeError(
                "Need to specify either pygid.Conversion instance or Nexus filename"
            )

        if self.pygid_conversion is not None and self.filename is not None:
            raise AttributeError(
                "Need to specify either pygid.Conversion instance or Nexus filename"
            )

    def _initialize_source(self) -> None:
        if self.pygid_conversion is not None:
            check_valid_conversion(self.pygid_conversion)
            self.from_nexus = False

        if self.filename is not None:
            self.entry_dict = get_entry_dict(self.filename)
            self.from_nexus = True
        ##test
        # peaks2img_container(self.filename, 'entry_0000', 0)


    def run_detection(self, entry = None, frame_num = None, config_detect = None):

        if config_detect is not None:
            if self.config_detect is not None:
                self.logger.info(f"config_detect is already set. The previous config is to be used")
            else:
                self.config_detect = config_detect
        self.config_detect = load_config(config_detect)
        if self.imp_detect is None:
            self.imp_detect = load_inference(self.config_detect)

        if not self.from_nexus:
            self.run_detection_from_memory()
        else:
            self.run_detection_from_file(entry, frame_num)

    def run_detection_from_memory(self):
        img_list = self.pygid_conversion.img_gid_q # if frame_num is None else [self.pygid_conversion.img_gid_q[frame_num]]
        q_xy = self.pygid_conversion.matrix[0].q_xy
        q_z = self.pygid_conversion.matrix[0].q_xy

        self.img_container_detect_list = []

        for img in img_list:
            img_container_detect = run_mlgiddetect(img, q_xy, q_z,
                                                             self.imp_detect, self.config_detect)
            self.img_container_detect_list.append(img_container_detect)

    def run_detection_from_file(self, entry, frame_num):
        if entry is None:
            for entry in self.entry_dict:
                self._run_detection_single_entry(entry, frame_num)
            return
        if not entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self._run_detection_single_entry(entry, frame_num)

    def _run_detection_single_entry(self, entry, frame_num):
        frame_num_all = self.entry_dict[entry]['shape'][0]
        if frame_num is None:
            for frame_num in range(frame_num_all):
                self._run_detection_single_frame(entry, frame_num)
            return
        if frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        self._run_detection_single_frame(entry, frame_num)

    def _run_detection_single_frame(self, entry, frame_num):
        conversion = read_conversion_from_nexus(self.filename, entry, frame_num)

        img_container_detect = run_mlgiddetect(conversion.img_gid_q[0], conversion.matrix[0].q_xy, conversion.matrix[0].q_z,
                         self.imp_detect, self.config_detect)
        save_detect(self.filename, entry, frame_num, img_container_detect)
        self.logger.info(f"Saved detected peaks to file: {self.filename}, entry: {entry}, frame: {frame_num}")


    def run_fitting(self, entry = None, frame_num = None, crit_angle = 0,
                    clustering_distance_peaks = 10, clustering_distance_rings = 10,
                    clustering_extend = 2, use_pool = False, debug = False, save_result = False):

        if not self.from_nexus:
            self.run_fitting_from_memory(clustering_distance_peaks= clustering_distance_peaks,
                                         clustering_distance_rings=clustering_distance_rings,
                                         clustering_extend=clustering_extend,
                                         use_pool=use_pool,
                                         debug=debug)
        else:
            run_pygidfit_from_file(filename = self.filename, entry=entry, frame_num = frame_num,
                                     crit_angle = crit_angle, polar_shape = np.array([512, 1024]),
                                     ratio_threshold = 50, clustering_distance_peaks = clustering_distance_peaks,
                                     clustering_distance_rings = clustering_distance_rings, clustering_extend = clustering_extend,
                                     use_pool = use_pool, debug = debug, multiprocessing = False)

    def run_fitting_from_memory(self,clustering_distance_peaks,
                                         clustering_distance_rings,
                                         clustering_extend,
                                         use_pool,
                                         debug):

        q_xy = self.pygid_conversion.matrix[0].q_xy
        q_z = self.pygid_conversion.matrix[0].q_xy
        wavelength = self.pygid_conversion.params.wavelength
        ang_deg_max = self.pygid_conversion.matrix[0].angular_range[-1]
        peaks_pool = [] if use_pool else None

        self.img_container_fit_list = []

        if self.img_container_detect_list is None:
            raise ValueError("img_container_detect_list is not defined. Call run_detection before run_fitting")
        for img_container_detect in self.img_container_detect_list:
            img_container_fit, peaks_pool = run_pygidfit_from_memory(img_container_detect = img_container_detect,
                                     wavelength = wavelength, q_xy_max = np.nanmax(q_xy), q_z_max = np.nanmax(q_z),
                                     ang_deg_max = ang_deg_max,
                                     ratio_threshold=50,
                                     clustering_distance_peaks = clustering_distance_peaks,
                                     clustering_distance_rings = clustering_distance_rings,
                                     clustering_extend = clustering_extend,
                                     peaks_pool = peaks_pool, debug = debug, multiprocessing = False)
            img_container_fit.q_xy = q_xy
            img_container_fit.q_z = q_z
            self.img_container_fit_list.append(img_container_fit)



    def run_matching(self, entry = None, frame_num = None, cif_prepr=None,
                     threshold=0.5, peaks_type='segments', device='cpu', save_result=False):

        if cif_prepr is not None:
            if self.cif_prepr is not None:
                self.logger.info(f"cif_prepr is already set. The previous cif_prepr is to be used")
            else:
                self.cif_prepr = cif_prepr
        self.cif_prepr = load_cif_prepr(self.cif_prepr)
        if not hasattr(self, 'match_class') or self.match_class is None:
            self.match_class = set_match_class(self.cif_prepr, device)
        if not self.from_nexus:
            self.run_matching_from_memory(threshold, peaks_type)
        else:
            self.run_matching_from_file(entry, frame_num, threshold, peaks_type)

    def run_matching_from_memory(self, threshold, peaks_type):
        self.container_match_list = []

        if self.img_container_fit_list is None:
            raise ValueError("img_container_fit_list is not defined. Call run_fitting before run_fitting")
        for frame_num, img_container_fit in enumerate(self.img_container_fit_list):
            q_xy_max = np.nanmax(img_container_fit.q_xy)
            q_z_max = np.nanmax(img_container_fit.q_z)
            is_ring = img_container_fit.is_ring
            amplitude = img_container_fit.amplitude
            q_z = img_container_fit.qzqxyboxes[0]
            q_xy = img_container_fit.qzqxyboxes[1]

            mask = is_ring if peaks_type == 'rings' else ~is_ring
            intensity_roi = amplitude[mask]
            q_2d_roi = np.column_stack((q_xy[mask], q_z[mask]))
            indices_roi = np.where(mask)[0]
            n_total = len(mask)

            unique_solutions = get_unique_solutions(self.match_class, peaks_type, threshold, q_xy_max, q_z_max, q_2d_roi, frame_num,
                                 intensity_roi, indices_roi, n_total)
            unique_solutions['peaks_type'] = peaks_type
            self.container_match_list.append(solution2container(unique_solutions))

    def run_matching_from_file(self, entry, frame_num, threshold, peaks_type):
        if entry is None:
            for entry in self.entry_dict:
                self._run_matching_single_entry(entry, frame_num, threshold, peaks_type)
            return
        if not entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self._run_matching_single_entry(entry, frame_num, threshold, peaks_type)

    def _run_matching_single_entry(self, entry, frame_num, threshold, peaks_type):
        frame_num_all = self.entry_dict[entry]['shape'][0]
        if frame_num is None:
            for frame_num in range(frame_num_all):
                self._run_matching_single_frame(entry, frame_num, threshold, peaks_type)
            return
        if frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        self._run_matching_single_frame(entry, frame_num, threshold, peaks_type)

    def _run_matching_single_frame(self, entry, frame_num, threshold, peaks_type):

        unique_solutions = run_mlgidmatch_from_file(self.filename, entry, frame_num, self.match_class,
                                                    threshold, peaks_type)
        if unique_solutions == {}:
            self.logger.info(f"No solutions for ({self.filename}, entry: {entry}, frame: {frame_num}) was found. "
                             f"Try to decrease threshold")
            return
        unique_solutions['peaks_type'] = peaks_type
        self.unique_solutions = unique_solutions
        # container_matched =  solution2container(unique_solutions)
        save_match(self.filename, entry, frame_num, solution2container(unique_solutions))
        self.logger.info(f"Saved matched peaks to file: {self.filename}, entry: {entry}, frame: {frame_num}")



    def run_pipeline(self, entry = None, frame_num = None,
                     config_file = None, config_detect = None, imp_detect = None,
                     crit_angle = 0,  clustering_extend = 2, clustering_distance_peaks = 10, clustering_distance_rings = 10,
                     use_pool = False, debug = False,
                     cif_prepr_file=None, cif_prepr=None,
                     threshold=0.5, peaks_type='segments', device='cpu'
                     ):

        self.update(
            config_file=config_file,
            config_detect=config_detect,
            imp_detect=imp_detect,
            crit_angle=crit_angle,
            clustering_distance_peaks=clustering_distance_peaks,
            clustering_distance_rings=clustering_distance_rings,
            clustering_extend=clustering_extend,
            use_pool=use_pool,
            debug=debug,
            cif_prepr_file=cif_prepr_file,
            cif_prepr=cif_prepr,
            threshold=threshold,
            peaks_type=peaks_type,
            device=device,
        )

        self.config_detect = self.config_detect or load_config(self.config_file)
        self.imp_detect = self.imp_detect or load_inference(self.config_detect)
        self.cif_prepr = self.cif_prepr or load_cif_prepr(self.cif_prepr_file)


        if not self.from_nexus:
            return
        else:
            self.run_pipeline_from_file(entry, frame_num, threshold, peaks_type)


    def run_pipeline_from_file(self, entry, frame_num, threshold, peaks_type):
        if entry is None:
            for entry in self.entry_dict:
                self._run_matching_single_entry(entry, frame_num)
            return
        if not entry in self.entry_dict:
            raise ValueError("entry not found in the NeXus file")
        self._run_pipeline_single_entry(entry, frame_num)

    def _run_pipeline_single_entry(self, entry, frame_num):
        frame_num_all = self.entry_dict[entry]['shape'][0]
        if frame_num is None:
            for frame_num in range(frame_num_all):
                self._run_pipeline_single_frame(entry, frame_num)
            return
        if frame_num >= frame_num_all:
            raise ValueError("frame_num is out of range")
        self._run_pipeline_single_frame(entry, frame_num)

    def save_result(self, path_to_save = 'result.h5', overwrite_file = True, h5_group = 'entry_0000',
                    overwrite_group = False, smpl_metadata = None, exp_metadata = None):


        save_pipeline(self.pygid_conversion, self.img_container_detect_list,
                      self.img_container_fit_list, self.container_match_list,
                      path_to_save, overwrite_file, h5_group, overwrite_group,
                      smpl_metadata, exp_metadata)

    def _run_pipeline_single_frame(self, entry, frame_num):
        return
    def plot_results(self): #TODO: add set plot defaults, symbols etc
        pass

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if not hasattr(self, key):
                raise AttributeError(f"Unknown parameter: {key}")
            setattr(self, key, value)