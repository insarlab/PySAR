############################################################
# Program is part of MintPy                                #
# Copyright (c) 2013, Zhang Yunjun, Heresh Fattahi         #
# Author: Zhang Yunjun, Heresh Fattahi, Jul 2013           #
############################################################


import glob
import os
import shutil
import time

import numpy as np

import mintpy
from mintpy.objects import RAMP_LIST, cluster, sensor
from mintpy.utils import readfile, utils as ut, writefile

# comment out the workflow import in favor of individual imports for speed and robustness
# import mintpy.workflow  # dynamic import of modules for smallbaselineApp


##########################################################################
def get_the_latest_default_template_file(work_dir):
    """Get the latest version of default template file.
    If an obsolete file exists in the working directory, the existing option values are kept.
    """
    # get the latest and current versions of the default template files
    lfile = os.path.join(os.path.dirname(mintpy.__file__), 'defaults/smallbaselineApp.cfg')
    cfile = os.path.join(work_dir, 'smallbaselineApp.cfg')

    if not os.path.isfile(cfile):
        print(f'copy default template file {lfile} to work directory')
        shutil.copy2(lfile, work_dir)
    else:
        #cfile is obsolete if any key is missing
        ldict = readfile.read_template(lfile)
        cdict = readfile.read_template(cfile)
        if any([key not in cdict.keys() for key in ldict.keys()]):
            print('obsolete default template detected, update to the latest version.')
            shutil.copy2(lfile, work_dir)

            #keep the existing option value from obsolete template file
            ut.update_template_file(cfile, cdict)

    return cfile


##########################################################################
class TimeSeriesAnalysis:
    """ Routine processing workflow for time series analysis of small baseline InSAR stacks
    """

    def __init__(self, customTemplateFile=None, workDir=None):
        self.customTemplateFile = customTemplateFile
        self.workDir = os.path.abspath(workDir)
        self.cwd = os.path.abspath(os.getcwd())

    def open(self):
        """The starting point of the workflow. It runs every time.
        It 1) grab project name if given
           2) go to work directory
           3) get and read template(s) options
        """

        #1. Get projectName
        self.projectName = None
        if self.customTemplateFile:
            self.projectName = os.path.splitext(os.path.basename(self.customTemplateFile))[0]
            print('Project name:', self.projectName)

        #2. Go to work directory
        os.makedirs(self.workDir, exist_ok=True)
        os.chdir(self.workDir)
        print('Go to work directory:', self.workDir)

        #3. Read templates
        self.templateFile = get_the_latest_default_template_file(self.workDir)
        self._read_template()


    def _read_template(self):
        # 1) update default template
        self.customTemplate = None
        if self.customTemplateFile:
            # customTemplateFile --> customTemplate
            print('read custom template file:', self.customTemplateFile)
            cdict = readfile.read_template(self.customTemplateFile)

            # correct some loose type errors
            standardValues = {
                'def':'auto', 'default':'auto',
                'y':'yes', 'on':'yes', 'true':'yes',
                'n':'no', 'off':'no', 'false':'no',
            }
            for key, value in cdict.items():
                if value in standardValues.keys():
                    cdict[key] = standardValues[value]

            for key in ['mintpy.deramp', 'mintpy.troposphericDelay.method']:
                if key in cdict.keys():
                    cdict[key] = cdict[key].lower().replace('-', '_')

            if 'processor' in cdict.keys():
                cdict['mintpy.load.processor'] = cdict['processor']

            # these metadata are used in load_data.py only, not needed afterwards
            # (in order to manually add extra offset when the lookup table is shifted)
            # (seen in ROI_PAC product sometimes)
            for key in ['SUBSET_XMIN', 'SUBSET_YMIN']:
                if key in cdict.keys():
                    cdict.pop(key)

            self.customTemplate = dict(cdict)

            # customTemplate --> templateFile
            print('update default template based on input custom template')
            self.templateFile = ut.update_template_file(self.templateFile, self.customTemplate)

        # 2) backup custom/default template file in inputs/pic folder
        flen = len(os.path.basename(self.templateFile))
        if self.customTemplateFile:
            flen = max(flen, len(os.path.basename(self.customTemplateFile)))

        for backup_dirname in ['inputs', 'pic']:
            backup_dir = os.path.join(self.workDir, backup_dirname)
            # create directory
            os.makedirs(backup_dir, exist_ok=True)

            # back up to the directory
            tfiles = [x for x in [self.customTemplateFile, self.templateFile] if x]
            for tfile in tfiles:
                out_file = os.path.join(backup_dir, os.path.basename(tfile))
                if ut.run_or_skip(out_file, in_file=tfile, readable=False, print_msg=False) == 'run':
                    shutil.copy2(tfile, backup_dir)
                    print('copy {f:<{l}} to {d:<8} directory for backup.'.format(
                        f=os.path.basename(tfile), l=flen, d=os.path.basename(backup_dir)))

        # 3) read default template file
        print('read default template file:', self.templateFile)
        self.template = readfile.read_template(self.templateFile)
        self.template = ut.check_template_auto_value(self.template)

        # correct some loose setup conflicts
        if self.template['mintpy.geocode'] is False:
            for key in ['mintpy.save.hdfEos5', 'mintpy.save.kmz']:
                if self.template[key] is True:
                    self.template['mintpy.geocode'] = True
                    print(f'Turn ON mintpy.geocode in order to run {key}.')
                    break


    def run_load_data(self, step_name):
        """Load InSAR stacks into HDF5 files in ./inputs folder.
        It 1) copy auxiliary files into work directory (for Unvi of Miami only)
           2) load all interferograms stack files into mintpy/inputs directory.
           3) check loading result
           4) add custom metadata (optional, for HDF-EOS5 format only)
        """
        # 1) copy aux files (optional)
        self._copy_aux_file()

        # 2) loading data
        # compose list of input arguments
        # instead of using command line then split
        # to support path with whitespace
        iargs = ['--template', self.templateFile]
        if self.customTemplateFile:
            iargs += [self.customTemplateFile]
        if self.projectName:
            iargs += ['--project', self.projectName]

        # run command line
        print('\nload_data.py', ' '.join(iargs))
        import mintpy.cli.load_data
        mintpy.cli.load_data.main(iargs)

        # come back to working directory
        os.chdir(self.workDir)

        # 3) check loading result
        stack_file, geom_file, _, ion_file = ut.check_loaded_dataset(self.workDir, print_msg=True)[:4]

        # 4) add custom metadata (optional)
        if self.customTemplateFile:
            # use ut.add_attribute() instead of add_attribute.py because of
            # better control of special metadata, such as SUBSET_X/YMIN
            msg = f'updating metadata based on custom template file {os.path.basename(self.customTemplateFile)}'
            for fname in [stack_file, ion_file]:  #, geom_file]:
                if fname:
                    print(f'{msg} for file: {os.path.basename(fname)}')
                    ut.add_attribute(fname, self.customTemplate)


    def _copy_aux_file(self):
        # for Univ of Miami
        if os.getenv('SCRATCHDIR') and self.projectName:
            proj_dir = os.path.join(os.getenv('SCRATCHDIR'), self.projectName)
            flist = ['PROCESS/unavco_attributes.txt',
                     'PROCESS/bl_list.txt',
                     'SLC/summary*slc.jpg']
            flist = ut.get_file_list([os.path.join(proj_dir, i) for i in flist], abspath=True)
            for fname in flist:
                if ut.run_or_skip(os.path.basename(fname), in_file=fname, readable=False) == 'run':
                    shutil.copy2(fname, self.workDir)
                    print(f'copy {os.path.basename(fname)} to work directory')


    def run_network_modification(self, step_name):
        """Modify network of interferograms before the network inversion."""
        # check the existence of ifgramStack.h5
        stack_file, geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[:2]
        coh_txt = os.path.join(self.workDir, 'coherenceSpatialAvg.txt')
        net_fig = [os.path.join(self.workDir, i, 'network.pdf') for i in ['', 'pic']]
        net_fig = [x for x in net_fig if os.path.isfile(x)]

        # 1) output waterMask.h5 to simplify the detection/use of waterMask
        water_mask_file = os.path.join(self.workDir, 'waterMask.h5')
        if 'waterMask' in readfile.get_dataset_list(geom_file):
            if ut.run_or_skip(out_file=water_mask_file, in_file=geom_file) == 'run':
                print(f'generate {water_mask_file} from {geom_file} for conveniency')
                water_mask, atr = readfile.read(geom_file, datasetName='waterMask')

                # ignore no-data pixels in geometry files
                ds_name_list = readfile.get_dataset_list(geom_file)
                for ds_name in ['latitude','longitude']:
                    if ds_name in ds_name_list:
                        print(f'set pixels with 0 in {ds_name} to 0 in waterMask')
                        ds = readfile.read(geom_file, datasetName=ds_name)[0]
                        water_mask[ds == 0] = 0

                atr['FILE_TYPE'] = 'waterMask'
                writefile.write(water_mask, out_file=water_mask_file, metadata=atr)

        # 2) modify network
        iargs = [stack_file, '-t', self.templateFile]
        print('\nmodify_network.py', ' '.join(iargs))
        import mintpy.cli.modify_network
        mintpy.cli.modify_network.main(iargs)

        # 3) plot network
        iargs = [stack_file, '-t', self.templateFile, '--nodisplay']

        dsNames = readfile.get_dataset_list(stack_file)
        if any('phase' in i.lower() for i in dsNames):
            iargs += ['-d', 'coherence', '-v', '0.2', '1.0']
        elif any('offset' in i.lower() for i in dsNames):
            iargs += ['-d', 'offsetSNR', '-v', '0', '20']

        print('\nplot_network.py', ' '.join(iargs))

        # run
        if self.template['mintpy.plot']:
            if ut.run_or_skip(out_file=net_fig,
                              in_file=[stack_file, coh_txt, self.templateFile],
                              readable=False) == 'run':
                import mintpy.cli.plot_network
                mintpy.cli.plot_network.main(iargs)
        else:
            print('mintpy.plot is turned OFF, skip plotting network.')


    def generate_ifgram_aux_file(self):
        """Generate auxiliary files from ifgramStack file"""
        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[0]
        dsNames = readfile.get_dataset_list(stack_file)
        mask_file = os.path.join(self.workDir, 'maskConnComp.h5')
        coh_file = os.path.join(self.workDir, 'avgSpatialCoh.h5')
        snr_file = os.path.join(self.workDir, 'avgSpatialSNR.h5')

        # 1) generate mask file from the common connected components
        if any('phase' in i.lower() for i in dsNames):
            iargs = [stack_file, '--nonzero', '-o', mask_file, '--update']
            print('\ngenerate_mask.py', ' '.join(iargs))
            import mintpy.cli.generate_mask
            mintpy.cli.generate_mask.main(iargs)

        # 2) generate average spatial coherence
        if any('phase' in i.lower() for i in dsNames):
            iargs = [stack_file, '--dataset', 'coherence', '-o', coh_file, '--update']
        elif any('offset' in i.lower() for i in dsNames):
            iargs = [stack_file, '--dataset', 'offsetSNR', '-o', snr_file, '--update']
        print('\ntemporal_average.py', ' '.join(iargs))
        import mintpy.cli.temporal_average
        mintpy.cli.temporal_average.main(iargs)


    def run_reference_point(self, step_name):
        """Select reference point.
        It 1) generate mask file from common conn comp
           2) generate average spatial coherence and its mask
           3) add REF_X/Y and/or REF_LAT/LON attribute to stack file
        """
        # 1-2) aux files: maskConnComp and avgSpatialCoh
        self.generate_ifgram_aux_file()

        # 3) add REF_X/Y(/LAT/LON) of the reference point
        stack_file, _, lookup_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[:3]
        coh_file = os.path.join(self.workDir, 'avgSpatialCoh.h5')

        iargs = [stack_file, '-t', self.templateFile, '-c', coh_file]
        if lookup_file is not None:
            iargs += ['--lookup', lookup_file]
        print('\nreference_point.py', ' '.join(iargs))
        import mintpy.cli.reference_point
        mintpy.cli.reference_point.main(iargs)


    def run_quick_overview(self, step_name):
        """A quick overview on the interferogram stack for:
            1) avgPhaseVelocity.h5: possible ground deformation through interferogram stacking
            2) numTriNonzeroIntAmbiguity.h5: phase unwrapping errors
                through the integer ambiguity of phase closure
        """
        # check the existence of ifgramStack.h5
        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[0]

        # 1) stack interferograms
        pha_vel_file = os.path.join(self.workDir, 'avgPhaseVelocity.h5')
        iargs = [stack_file, '--dataset', 'unwrapPhase', '-o', pha_vel_file, '--update']
        print('\ntemporal_average.py', ' '.join(iargs))
        import mintpy.cli.temporal_average
        mintpy.cli.temporal_average.main(iargs)

        # 2) calculate the number of interferogram triplets with non-zero integer ambiguity
        water_mask_file = os.path.join(self.workDir, 'waterMask.h5')
        iargs = [stack_file, '--water-mask', water_mask_file, '--action', 'calculate', '--update']
        print('\nunwrap_error_phase_closure.py', ' '.join(iargs))
        import mintpy.cli.unwrap_error_phase_closure
        mintpy.cli.unwrap_error_phase_closure.main(iargs)


    def run_unwrap_error_correction(self, step_name):
        """Correct phase-unwrapping errors"""
        method = self.template['mintpy.unwrapError.method']
        if not method:
            print('phase-unwrapping error correction is OFF.')
            return

        # check required input files
        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[0]
        mask_file = os.path.join(self.workDir, 'maskConnComp.h5')

        iargs_bridge = [stack_file, '--template', self.templateFile, '--update']
        iargs_closure = iargs_bridge + ['--cc-mask', mask_file]

        import mintpy.cli.unwrap_error_bridging
        import mintpy.cli.unwrap_error_phase_closure
        if method == 'bridging':
            print('\nunwrap_error_bridging.py', ' '.join(iargs_bridge))
            mintpy.cli.unwrap_error_bridging.main(iargs_bridge)

        elif method == 'phase_closure':
            print('\nunwrap_error_phase_closure.py', ' '.join(iargs_closure))
            mintpy.cli.unwrap_error_phase_closure.main(iargs_closure)

        elif method == 'bridging+phase_closure':
            iargs_bridge += ['-i', 'unwrapPhase',
                             '-o', 'unwrapPhase_bridging']
            print('\nunwrap_error_bridging.py', ' '.join(iargs_bridge))
            mintpy.cli.unwrap_error_bridging.main(iargs_bridge)

            iargs_closure += ['-i', 'unwrapPhase_bridging',
                              '-o', 'unwrapPhase_bridging_phaseClosure']
            print('\nunwrap_error_phase_closure.py', ' '.join(iargs_closure))
            mintpy.cli.unwrap_error_phase_closure.main(iargs_closure)

        else:
            raise ValueError(f'un-recognized method: {method}')


    def run_network_inversion(self, step_name):
        """Invert network of interferograms for raw phase time-series.
        1) network inversion --> timeseries.h5, temporalCoherence.h5, numInvIfgram.h5
        2) temporalCoherence.h5 --> maskTempCoh.h5
        """
        # check the existence of ifgramStack.h5
        stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[0]

        # 1) invert ifgramStack for time-series
        iargs = [stack_file, '-t', self.templateFile, '--update']
        print('\nifgram_inversion.py', ' '.join(iargs))
        import mintpy.cli.ifgram_inversion
        mintpy.cli.ifgram_inversion.main(iargs)

        # 2) get reliable pixel mask: maskTempCoh.h5
        self.generate_temporal_coherence_mask()


    def generate_temporal_coherence_mask(self):
        """Generate reliable pixel mask from temporal coherence"""
        geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        tcoh_file = os.path.join(self.workDir, 'temporalCoherence.h5')
        mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')
        tcoh_min = self.template['mintpy.networkInversion.minTempCoh']

        # compose list of arguments
        iargs = [tcoh_file, '-m', tcoh_min, '-o', mask_file]
        # exclude pixels in shadow if shadowMask dataset is available
        if (self.template['mintpy.networkInversion.shadowMask'] is True
                and 'shadowMask' in readfile.get_dataset_list(geom_file)):
            iargs += ['--base', geom_file, '--base-dataset', 'shadowMask', '--base-value', '1']
        print('\ngenerate_mask.py', ' '.join(iargs))

        # update mode: run only if:
        # 1) output file exists and newer than input file, AND
        # 2) all config keys are the same
        config_keys = [f'mintpy.networkInversion.{i}' for i in ['minTempCoh','shadowMask']]
        print('update mode: ON')
        flag = 'skip'
        if ut.run_or_skip(out_file=mask_file, in_file=tcoh_file, print_msg=False) == 'run':
            flag = 'run'
        else:
            print(f'1) output file: {mask_file} already exists and newer than input file: {tcoh_file}')
            atr = readfile.read_attribute(mask_file)
            if any(str(self.template[i]) != atr.get(i, 'False') for i in config_keys):
                flag = 'run'
                print(f'2) NOT all key configuration parameters are the same: {config_keys}')
            else:
                print(f'2) all key configuration parameters are the same: {config_keys}')
        print(f'run or skip: {flag}')

        if flag == 'run':
            import mintpy.cli.generate_mask
            mintpy.cli.generate_mask.main(iargs)

            # update config_keys
            atr = {}
            for key in config_keys:
                atr[key] = self.template[key]
            ut.add_attribute(mask_file, atr)

        # check number of pixels selected in mask file for following analysis
        num_pixel = np.sum(readfile.read(mask_file)[0] != 0.)
        print(f'number of reliable pixels: {num_pixel}')

        min_num_pixel = float(self.template['mintpy.networkInversion.minNumPixel'])
        if num_pixel < min_num_pixel:
            msg = f"Not enough reliable pixels (minimum of {int(min_num_pixel)}). "
            msg += "Try the following:\n"
            msg += "1) Check the reference pixel and make sure it's not in areas with unwrapping errors\n"
            msg += "2) Check the network and make sure it's fully connected without subsets"
            print(f'ERROR: {msg}')

            # populate the pic folder to facilate the trouble shooting
            self.plot_result(print_aux=False)

            # terminate the program
            raise RuntimeError(msg)


    @staticmethod
    def get_timeseries_filename(template, work_dir='./'):
        """Get input/output time-series filename for each step
        Parameters: template : dict, content of smallbaselineApp.cfg
        Returns:    steps    : dict of dicts, input/output filenames for each step
        """
        work_dir = os.path.abspath(work_dir)
        fname0 = os.path.join(work_dir, 'timeseries.h5')
        fname1 = os.path.join(work_dir, 'timeseries.h5')
        #fname0 = os.path.join(work_dir, 'timeseries_ramp_demErr.h5')
        #fname1 = os.path.join(work_dir, 'timeseries_ramp_demErr.h5')
        atr = readfile.read_attribute(fname0)

        phase_correction_steps = [
            'correct_LOD',
            'correct_SET',
            'correct_ionosphere',
            'correct_troposphere',
            'deramp',
            'correct_topography',
        ]

        # loop for all steps
        steps = dict()
        for sname in phase_correction_steps:
            # fname0 == fname1 if no valid correction method is set.
            fname0 = fname1
            if sname == 'correct_LOD':
                if atr['PLATFORM'].lower().startswith('env'):
                    fname1 = f'{os.path.splitext(fname0)[0]}_LOD.h5'

            elif sname == 'correct_SET':
                method = template['mintpy.solidEarthTides']
                if method:
                    fname1 = f'{os.path.splitext(fname0)[0]}_SET.h5'

            elif sname == 'correct_ionosphere':
                method = template['mintpy.ionosphericDelay.method']
                if method:
                    if method == 'split_spectrum':
                        fname1 = f'{os.path.splitext(fname0)[0]}_ion.h5'
                    else:
                        msg = f'un-recognized ionospheric correction method: {method}'
                        raise ValueError(msg)

            elif sname == 'correct_troposphere':
                method = template['mintpy.troposphericDelay.method']
                model  = template['mintpy.troposphericDelay.weatherModel']
                if method:
                    if method == 'height_correlation':
                        fname1 = f'{os.path.splitext(fname0)[0]}_tropHgt.h5'

                    elif method == 'gacos':
                        fname1 = f'{os.path.splitext(fname0)[0]}_GACOS.h5'

                    elif method == 'pyaps':
                        fname1 = f'{os.path.splitext(fname0)[0]}_{model}.h5'

                    elif method == 'local_texture':
                        fname1 = f'{os.path.splitext(fname0)[0]}_tropolocaltexture.h5'

                    else:
                        msg = f'un-recognized tropospheric correction method: {method}'
                        raise ValueError(msg)

            elif sname == 'deramp':
                method = template['mintpy.deramp']
                if method:
                    if method in RAMP_LIST:
                        fname1 = f'{os.path.splitext(fname0)[0]}_ramp.h5'
                    else:
                        msg = f'un-recognized phase ramp type: {method}'
                        msg += f'\navailable ramp types:\n{RAMP_LIST}'
                        raise ValueError(msg)

            elif sname == 'correct_topography':
                method = template['mintpy.topographicResidual']
                if method:
                    fname1 = f'{os.path.splitext(fname0)[0]}_demErr.h5'

            step = dict()
            step['input'] = fname0
            step['output'] = fname1
            steps[sname] = step

        # step - reference_date
        fnames = [steps[sname]['output'] for sname in phase_correction_steps]
        fnames += [steps[sname]['input'] for sname in phase_correction_steps]
        fnames = sorted(list(set(fnames)))
        step = dict()
        step['input'] = fnames
        steps['reference_date'] = step

        # step - velocity / geocode
        step = dict()
        step['input'] = steps['reference_date']['input'][-1]
        steps['velocity'] = step
        steps['geocode'] = step

        # step - hdfeos5
        if 'Y_FIRST' not in atr.keys():
            step = dict()
            fdir = os.path.dirname(steps['reference_date']['input'][-1])
            fbase = os.path.basename(steps['reference_date']['input'][-1])
            step['input'] = os.path.join(fdir, f'geo/geo_{fbase}')
        steps['hdfeos5'] = step
        return steps


    def run_local_oscillator_drift_correction(self, step_name):
        """Correct local oscillator drift (LOD).
        Automatically applied for Envisat data.
        Automatically skipped for all the other data.
        """
        geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        fnames = self.get_timeseries_filename(self.template, self.workDir)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']
        if in_file != out_file:
            iargs = [in_file, geom_file, '-o', out_file]
            print('\nlocal_oscilator_drift.py', ' '.join(iargs))
            if ut.run_or_skip(out_file=out_file, in_file=in_file) == 'run':
                import mintpy.cli.local_oscilator_drift
                mintpy.cli.local_oscilator_drift.main(iargs)
        else:
            atr = readfile.read_attribute(in_file)
            sat = atr.get('PLATFORM', None)
            print(f'No local oscillator drift correction is needed for {sat}.')


    def run_solid_earth_tides_correction(self, step_name):
        """Correct solid Earth tides (SET)."""
        geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        fnames = self.get_timeseries_filename(self.template, self.workDir)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']

        if in_file != out_file:
            iargs = [in_file, '-g', geom_file, '-o', out_file, '--update']
            print('\nsolid_earth_tides.py', ' '.join(iargs))
            if ut.run_or_skip(out_file=out_file, in_file=in_file) == 'run':
                import mintpy.cli.solid_earth_tides
                mintpy.cli.solid_earth_tides.main(iargs)
        else:
            print('No solid Earth tides correction.')


    def run_ionospheric_delay_correction(self, step_name):
        """Correct ionospheric delays."""
        iono_stack_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[3]

        fnames = self.get_timeseries_filename(self.template, self.workDir)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']
        if in_file != out_file:
            method = self.template['mintpy.ionosphericDelay.method']

            # range split spectrum (Fattahi et al., 2017; Liang et al. 2018; 2019)
            if method == 'split_spectrum':
                print(f'ionospheric delay correction with {method} approach')
                iargs = ['-t', self.templateFile, '-f', in_file, '-o', out_file,
                         '--iono-stack-file', iono_stack_file]
                print('\niono_split_spectrum.py', ' '.join(iargs))
                import mintpy.cli.iono_split_spectrum
                mintpy.cli.iono_split_spectrum.main(iargs)
        else:
            print('No ionospheric delay correction.')


    def run_tropospheric_delay_correction(self, step_name):
        """Correct tropospheric delays."""
        geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')
        #For test only
        #geom_file = os.path.join(self.workDir, 'inputs/geometryGeo.h5')
        #mask_file = os.path.join(self.workDir, 'temporalCoherence.h5')

        fnames = self.get_timeseries_filename(self.template, self.workDir)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']
        if in_file != out_file:
            poly_order  = self.template['mintpy.troposphericDelay.polyOrder']
            tropo_model = self.template['mintpy.troposphericDelay.weatherModel'].upper()
            weather_dir = self.template['mintpy.troposphericDelay.weatherDir']
            method      = self.template['mintpy.troposphericDelay.method']

            def get_dataset_size(fname):
                atr = readfile.read_attribute(fname)
                return (atr['LENGTH'], atr['WIDTH'])

            # Phase/Elevation Ratio (Doin et al., 2009)
            if method == 'height_correlation':
                tropo_look = self.template['mintpy.troposphericDelay.looks']
                tropo_min_cor = self.template['mintpy.troposphericDelay.minCorrelation']
                iargs = [in_file,
                         '-g', geom_file,
                         '-p', poly_order,
                         '-m', mask_file,
                         '-o', out_file,
                         '-l', tropo_look,
                         '-t', tropo_min_cor]
                print('tropospheric delay correction with height-correlation approach')
                print('\ntropo_phase_elevation.py', ' '.join(iargs))
                if ut.run_or_skip(out_file=out_file, in_file=in_file) == 'run':
                    import mintpy.cli.tropo_phase_elevation
                    mintpy.cli.tropo_phase_elevation.main(iargs)

            # Weather re-analysis data with iterative tropospheric decomposition (GACOS)
            # Yu et al. (2018, JGR)
            elif method == 'gacos':
                GACOS_dir = self.template['mintpy.troposphericDelay.gacosDir']
                iargs = ['-f', in_file, '-g', geom_file, '-o', out_file, '--dir', GACOS_dir]
                print('tropospheric delay correction with gacos approach')
                print('\ntropo_gacos.py', ' '.join(iargs))
                if ut.run_or_skip(out_file=out_file, in_file=in_file) == 'run':
                    import mintpy.cli.tropo_gacos
                    mintpy.cli.tropo_gacos.main(iargs)

            # Weather Re-analysis Data (Jolivet et al., 2011;2014)
            elif method == 'pyaps':
                iargs = ['-f', in_file, '--model', tropo_model, '-g', geom_file, '-w', weather_dir]
                print('Atmospheric correction using Weather Re-analysis dataset (PyAPS, Jolivet et al., 2011)')
                print('Weather Re-analysis dataset:', tropo_model)
                tropo_file = f'./inputs/{tropo_model}.h5'

                if ut.run_or_skip(out_file=out_file, in_file=[in_file, tropo_file]) == 'run':
                    if os.path.isfile(tropo_file) and get_dataset_size(tropo_file) == get_dataset_size(in_file):
                        iargs = [in_file, tropo_file, '-o', out_file, '--force']
                        print('--------------------------------------------')
                        print(f'Use existed tropospheric delay file: {tropo_file}')
                        print('\ndiff.py', ' '.join(iargs))
                        import mintpy.cli.diff
                        mintpy.cli.diff.main(iargs)

                    else:
                        if tropo_model in ['ERA5']:
                            print('\ntropo_pyaps3.py', ' '.join(iargs))
                            import mintpy.cli.tropo_pyaps3
                            mintpy.cli.tropo_pyaps3.main(iargs)

                        elif tropo_model in ['MERRA', 'NARR']:
                            print('\ntropo_pyaps.py', ' '.join(iargs))
                            import mintpy.legacy.tropo_pyaps
                            mintpy.legacy.tropo_pyaps.main(iargs)

                        else:
                            raise ValueError(f'un-recognized dataset name: {tropo_model}.')
            # High-frequency Texture Correction (Yang et al., 2024)
            elif method == 'local_texture':
                window_size = self.template['mintpy.troposphericDelay.windowSize']
                overlap_ratio = self.template['mintpy.troposphericDelay.overlapRatio']
                iargs = [in_file, '-g', geom_file, '-m', mask_file, '-o', out_file, '-w' , window_size, '-r', overlap_ratio]
                print('tropospheric delay correction with high frequency texture correlation approach')
                print('\ntropo_local_texture.py', ' '.join(iargs))
                if ut.run_or_skip(out_file=out_file, in_file=in_file) == 'run':
                    import mintpy.cli.tropo_local_texture
                    mintpy.cli.tropo_local_texture.main(iargs)

        else:
            print('No tropospheric delay correction.')


    def run_phase_deramping(self, step_name):
        """Estimate and remove phase ramp from each acquisition."""
        mask_file = self.template['mintpy.deramp.maskFile']
        method    = self.template['mintpy.deramp']

        fnames = self.get_timeseries_filename(self.template, self.workDir)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']
        if in_file != out_file:
            print(f'Remove for each acquisition a phase ramp: {method}')
            iargs = [in_file, '-s', method, '-m', mask_file, '-o', out_file, '--update']
            print('\nremove_ramp.py', ' '.join(iargs))
            import mintpy.cli.remove_ramp
            mintpy.cli.remove_ramp.main(iargs)
        else:
            print('No phase ramp removal.')


    def run_topographic_residual_correction(self, step_name):
        """step - correct_topography
        Topographic residual (DEM error) correction (optional).
        """
        geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
        fnames = self.get_timeseries_filename(self.template, self.workDir)[step_name]
        in_file = fnames['input']
        out_file = fnames['output']

        if in_file != out_file:
            iargs = [in_file, '-t', self.templateFile, '-o', out_file, '--update']
            if self.template['mintpy.topographicResidual.pixelwiseGeometry']:
                iargs += ['-g', geom_file]
            print('\ndem_error.py', ' '.join(iargs))
            import mintpy.cli.dem_error
            mintpy.cli.dem_error.main(iargs)

        else:
            print('No topographic residual correction.')


    def run_residual_phase_rms(self, step_name):
        """Noise evaluation based on the phase residual."""
        res_file = 'timeseriesResidual.h5'
        if os.path.isfile(res_file):
            iargs = [res_file, '-t', self.templateFile]
            print('\ntimeseries_rms.py', ' '.join(iargs))
            import mintpy.cli.timeseries_rms
            mintpy.cli.timeseries_rms.main(iargs)
        else:
            print('No residual phase file found! Skip residual RMS analysis.')


    def run_reference_date(self, step_name):
        """Change reference date for all time-series files (optional)."""
        if self.template['mintpy.reference.date']:
            iargs = ['-t', self.templateFile]
            in_files = self.get_timeseries_filename(self.template, self.workDir)[step_name]['input']
            for in_file in in_files:
                iargs += [in_file]
            print('\nreference_date.py', ' '.join(iargs))
            import mintpy.cli.reference_date
            mintpy.cli.reference_date.main(iargs)
        else:
            print('No reference date change.')


    def run_timeseries2velocity(self, step_name):
        """Estimate average velocity from displacement time-series"""
        ts_file = self.get_timeseries_filename(self.template, self.workDir)[step_name]['input']
        vel_file = os.path.join(self.workDir, 'velocity.h5')

        iargs = [ts_file, '-t', self.templateFile, '-o', vel_file, '--update']
        print('\ntimeseries2velocity.py', ' '.join(iargs))
        import mintpy.cli.timeseries2velocity
        mintpy.cli.timeseries2velocity.main(iargs)

        # Velocity from estimated tropospheric delays
        tropo_model = self.template['mintpy.troposphericDelay.weatherModel'].upper()
        tropo_file = os.path.join(self.workDir, f'inputs/{tropo_model}.h5')
        if os.path.isfile(tropo_file):
            suffix = os.path.splitext(os.path.basename(tropo_file))[0]
            tropo_vel_file = f'{os.path.splitext(vel_file)[0]}{suffix}.h5'
            tropo_vel_file = os.path.join(self.workDir, tropo_vel_file)

            iargs = [tropo_file, '-t', self.templateFile, '-o', tropo_vel_file, '--update']
            # add reference info for a meaningful velocity to assess the impact of tropo delay on velocity
            atr = readfile.read_attribute(vel_file)
            iargs += ['--ref-date', atr['REF_DATE'], '--ref-yx', atr['REF_Y'], atr['REF_X']]
            print('\ntimeseries2velocity.py', ' '.join(iargs))
            mintpy.cli.timeseries2velocity.main(iargs)


    def run_geocode(self, step_name):
        """geocode data files in radar coordinates into ./geo folder."""
        if self.template['mintpy.geocode']:
            ts_file = self.get_timeseries_filename(self.template, self.workDir)[step_name]['input']
            atr = readfile.read_attribute(ts_file)
            if 'Y_FIRST' not in atr.keys():
                # 1. geocode
                out_dir = os.path.join(self.workDir, 'geo')
                os.makedirs(out_dir, exist_ok=True)

                geom_file, lookup_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1:3]
                in_files = [geom_file, 'temporalCoherence.h5', 'avgSpatialCoh.h5', ts_file, 'velocity.h5']
                iargs = in_files + ['-l', lookup_file, '-t', self.templateFile, '--outdir', out_dir, '--update']
                print('\ngeocode.py', ' '.join(iargs))
                import mintpy.cli.geocode
                mintpy.cli.geocode.main(iargs)

                # 2. generate reliable pixel mask in geo coordinate
                geom_file = os.path.join(out_dir, f'geo_{os.path.basename(geom_file)}')
                tcoh_file = os.path.join(out_dir, 'geo_temporalCoherence.h5')
                mask_file = os.path.join(out_dir, 'geo_maskTempCoh.h5')
                tcoh_min = self.template['mintpy.networkInversion.minTempCoh']

                iargs = [tcoh_file, '-m', tcoh_min, '-o', mask_file]
                # exclude pixels in shadow if shadowMask dataset is available
                if (self.template['mintpy.networkInversion.shadowMask'] is True
                        and 'shadowMask' in readfile.get_dataset_list(geom_file)):
                    iargs += ['--base', geom_file, '--base-dataset', 'shadowMask', '--base-value', '1']
                print('\ngenerate_mask.py', ' '.join(iargs))

                if ut.run_or_skip(out_file=mask_file, in_file=tcoh_file) == 'run':
                    import mintpy.cli.generate_mask
                    mintpy.cli.generate_mask.main(iargs)

            else:
                print('dataset is geocoded, skip geocoding and continue.')
        else:
            print('geocoding is OFF')


    def run_save2google_earth(self, step_name):
        """Save velocity file in geo coordinates into Google Earth raster image."""
        if self.template['mintpy.save.kmz'] is True:
            print('creating Google Earth KMZ file for geocoded velocity file: ...')
            # input
            vel_file = os.path.join(self.workDir, 'velocity.h5')
            atr = readfile.read_attribute(vel_file)
            if 'Y_FIRST' not in atr.keys():
                vel_file = os.path.join(self.workDir, 'geo/geo_velocity.h5')

            # output
            kmz_file = f'{os.path.splitext(vel_file)[0]}.kmz'
            iargs = [vel_file, '-o', kmz_file]
            print('\nsave_kmz.py', ' '.join(iargs))

            # update mode
            fbase = os.path.basename(kmz_file)
            kmz_files = [i for i in [fbase, f'./geo/{fbase}', f'./pic/{fbase}']
                         if os.path.isfile(i)]
            kmz_file = kmz_files[0] if len(kmz_files) > 0 else None

            if ut.run_or_skip(out_file=kmz_file, in_file=vel_file, readable=False) == 'run':
                import mintpy.cli.save_kmz
                mintpy.cli.save_kmz.main(iargs)

        else:
            print('save velocity to Google Earth format is OFF.')


    def run_save2hdfeos5(self, step_name):
        """Save displacement time-series and its aux data in geo coordinate into HDF-EOS5 format"""
        if self.template['mintpy.save.hdfEos5'] is True:
            # input
            ts_file = self.get_timeseries_filename(self.template, self.workDir)[step_name]['input']
            # Add attributes from custom template to timeseries file
            if self.customTemplate is not None:
                ut.add_attribute(ts_file, self.customTemplate)

            tcoh_file = os.path.join(self.workDir, 'temporalCoherence.h5')
            scoh_file = os.path.join(self.workDir, 'avgSpatialCoh.h5')
            mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')
            geom_file = ut.check_loaded_dataset(self.workDir, print_msg=False)[1]
            if 'geo' in ts_file:
                tcoh_file = os.path.join(self.workDir, 'geo/geo_temporalCoherence.h5')
                scoh_file = os.path.join(self.workDir, 'geo/geo_avgSpatialCoh.h5')
                mask_file = os.path.join(self.workDir, 'geo/geo_maskTempCoh.h5')
                geom_file = os.path.join(self.workDir, f'geo/geo_{os.path.basename(geom_file)}')

            # cmd
            print('--------------------------------------------')
            iargs = [ts_file,
                     '--tc', tcoh_file,
                     '--asc', scoh_file,
                     '-m', mask_file,
                     '-g', geom_file,
                     '-t', self.templateFile]
            print('\nsave_hdfeos5.py', ' '.join(iargs))

            # output (check existing file)
            atr = readfile.read_attribute(ts_file)
            SAT = sensor.get_unavco_mission_name(atr)
            hdfeos5_files = ut.get_file_list(f'{SAT}_*.he5')
            hdfeos5_file = hdfeos5_files[0] if len(hdfeos5_files) > 0 else None

            if ut.run_or_skip(out_file=hdfeos5_file,
                              in_file=[ts_file, tcoh_file,
                                       scoh_file, mask_file,
                                       geom_file]) == 'run':
                import mintpy.cli.save_hdfeos5
                mintpy.cli.save_hdfeos5.main(iargs)

        else:
            print('save time-series to HDF-EOS5 format is OFF.')


    def run(self, steps):
        """run the chosen steps."""
        for sname in steps:
            print(f'\n\n******************** step - {sname} ********************')

            if sname == 'load_data':
                self.run_load_data(sname)

            elif sname == 'modify_network':
                self.run_network_modification(sname)

            elif sname == 'reference_point':
                self.run_reference_point(sname)

            elif sname == 'quick_overview':
                self.run_quick_overview(sname)

            elif sname == 'correct_unwrap_error':
                self.run_unwrap_error_correction(sname)

            elif sname == 'invert_network':
                self.run_network_inversion(sname)

            elif sname == 'correct_LOD':
                self.run_local_oscillator_drift_correction(sname)

            elif sname == 'correct_SET':
                self.run_solid_earth_tides_correction(sname)

            elif sname == 'correct_ionosphere':
                self.run_ionospheric_delay_correction(sname)

            elif sname == 'correct_troposphere':
                self.run_tropospheric_delay_correction(sname)

            elif sname == 'deramp':
                self.run_phase_deramping(sname)

            elif sname == 'correct_topography':
                self.run_topographic_residual_correction(sname)

            elif sname == 'residual_RMS':
                self.run_residual_phase_rms(sname)

            elif sname == 'reference_date':
                self.run_reference_date(sname)

            elif sname == 'velocity':
                self.run_timeseries2velocity(sname)

            elif sname == 'geocode':
                self.run_geocode(sname)

            elif sname == 'google_earth':
                self.run_save2google_earth(sname)

            elif sname == 'hdfeos5':
                self.run_save2hdfeos5(sname)


    def plot_result(self, print_aux=True):
        """Plot data files and save to figures in pic folder"""

        print('\n******************** plot & save to pic ********************')

        tropo_model = self.template['mintpy.troposphericDelay.weatherModel'].upper()
        max_plot_memory = abs(float(self.template['mintpy.plot.maxMemory']))
        max_memory = abs(float(self.template['mintpy.compute.maxMemory']))
        fig_dpi = int(self.template['mintpy.plot.dpi'])

        stack_file, geom_file, lookup_file, ion_file = ut.check_loaded_dataset(
            self.workDir,
            print_msg=False)[:4]
        mask_file = os.path.join(self.workDir, 'maskTempCoh.h5')
        geo_dir = os.path.join(self.workDir, 'geo')
        pic_dir = os.path.join(self.workDir, 'pic')

        # use relative path for shorter and cleaner printout view command
        stack_file  = os.path.relpath(stack_file)  if stack_file  else stack_file
        ion_file    = os.path.relpath(ion_file)    if ion_file    else ion_file
        geom_file   = os.path.relpath(geom_file)   if geom_file   else geom_file
        lookup_file = os.path.relpath(lookup_file) if lookup_file else lookup_file
        mask_file   = os.path.relpath(mask_file)   if mask_file   else mask_file
        geo_dir     = os.path.relpath(geo_dir)     if geo_dir     else geo_dir

        # view options
        # for each element list:
        # the 1st item is the data file
        # the 2nd item is the dataset if applicable
        opt4ts = ['--noaxis', '-u', 'cm', '--wrap', '--wrap-range', '-5', '5']
        iargs_list0 = [
            # key files
            ['velocity.h5',          '--dem', geom_file, '--mask', mask_file],
            ['temporalCoherence.h5', '-c', 'gray', '-v', '0', '1'],
            ['maskTempCoh.h5',       '-c', 'gray', '-v', '0', '1'],

            # geometry
            [geom_file],
            [lookup_file],

            # ifgramStack
            [stack_file, 'unwrapPhase-',      '--zero-mask', '--wrap', '-c', 'cmy'],
            [stack_file, 'unwrapPhase-',      '--zero-mask'],
            [stack_file, 'coherence-',        '--mask', 'no', '-v', '0', '1'],
            [stack_file, 'connectComponent-', '--mask', 'no'],

            # ifgramStack - unwrapping error correction
            [stack_file, 'unwrapPhase_bridging-',              '--zero-mask'],
            [stack_file, 'unwrapPhase_phaseClosure-',          '--zero-mask'],
            [stack_file, 'unwrapPhase_bridging_phaseClosure-', '--zero-mask'],

            # ifgramStack - auxliary files
            ['avgPhaseVelocity.h5'] ,
            ['avgSpatialCoh.h5', '-c', 'gray', '-v', '0', '1'],
            ['maskConnComp.h5',  '-c', 'gray', '-v', '0', '1'],

            # time-series
            ['timeseries.h5'] + opt4ts,
            ['timeseries_*.h5'] + opt4ts,

            # files from geocoding
            [os.path.join(geo_dir, 'geo_maskTempCoh.h5'),       '-c', 'gray'],
            [os.path.join(geo_dir, 'geo_temporalCoherence.h5'), '-c', 'gray'],
            [os.path.join(geo_dir, 'geo_avgSpatialCoh.h5'),     '-c', 'gray'],
            [os.path.join(geo_dir, 'geo_velocity.h5'),          'velocity'],
            [os.path.join(geo_dir, 'geo_timeseries*.h5')] + opt4ts,

            # all the other files
            [f'velocity{tropo_model}.h5', '--mask', 'no'],
            ['numInvIfgram.h5',           '--mask', 'no'],
        ]

        if ion_file:
            iargs_list0 += [
                [ion_file, 'unwrapPhase-', '--zero-mask', '--wrap', '-c', 'cmy'],
                [ion_file, 'unwrapPhase-', '--zero-mask'],
                [ion_file, 'coherence-',   '--mask', 'no', '-v', '0', '1'],
            ]

        # translate element list whose file path has *
        iargs_list = []
        for iargs in iargs_list0:
            fname, args = iargs[0], iargs[1:]
            if not fname:
                continue

            if '*' in fname:
                fnames = sorted(glob.glob(fname))
                if len(fnames) > 0:
                    for fname in fnames:
                        iargs_list.append([fname] + args)

            elif iargs not in iargs_list:
                iargs_list.append(iargs)

        # remove element list - file does not exists
        iargs_list = [iargs for iargs in iargs_list if os.path.isfile(iargs[0])]

        # remote element list - file is ifgramStack and the dataset of interest does not exist
        stack_dset_list = readfile.get_dataset_list(stack_file)
        stack_dset_list += [f'{x}-' for x in stack_dset_list]
        iargs_list = [iargs for iargs in iargs_list
                      if (iargs[0] != stack_file
                          or (iargs[0] == stack_file
                              and iargs[1] in stack_dset_list))]

        # add the following common options to all element lists
        opt_common = ['--dpi', str(fig_dpi), '--noverbose', '--nodisplay',
                      '--update', '--memory', str(max_plot_memory)]
        iargs_list = [opt_common + iargs for iargs in iargs_list]

        # run view
        start_time = time.time()
        run_parallel = False
        cluster_type = self.template['mintpy.compute.cluster']
        if cluster_type:
            num_workers = self.template['mintpy.compute.numWorker']
            num_workers = cluster.DaskCluster.format_num_worker(cluster_type, num_workers)

            # limit number of parallel processes based on available CPU
            num_cores, run_parallel, Parallel, delayed = ut.check_parallel(
                len(iargs_list),
                print_msg=False,
                maxParallelNum=num_workers)

            # limit number of parallel processes based on memory limit
            # default view.py call could use up to 1.5 GB reserved memory (~700 MB actual memory)
            # scale based on utils.plot.auto_multilook_num()
            plot_memory = 1.5 if 2.0 < max_plot_memory <= 4.0 else 1.5 * (max_plot_memory / 4.0)
            num_cores = min(num_cores, max(int(max_memory / plot_memory), 1))

        import mintpy.cli.view
        if run_parallel and num_cores > 1:
            print(f"parallel processing using {num_cores} cores ...")
            Parallel(n_jobs=num_cores)(delayed(mintpy.cli.view.main)(iargs) for iargs in iargs_list)
        else:
            for iargs in iargs_list:
                mintpy.cli.view.main(iargs)

        # copy text files to pic
        print('copy *.txt files into ./pic directory.')
        tfiles = glob.glob('*.txt')
        for tfile in tfiles:
            shutil.copy2(tfile, pic_dir)

        # move picture files to pic
        print('move *.png/pdf/kmz files to ./pic directory.')
        pfiles  = glob.glob('*.png')
        pfiles += glob.glob('*.pdf')
        pfiles += glob.glob('*.kmz')
        pfiles += glob.glob(os.path.join(geo_dir, '*.kmz'))
        for pfile in pfiles:
            shutil.move(pfile, os.path.join(pic_dir, os.path.basename(pfile)))

        # time info
        m, s = divmod(time.time()-start_time, 60)
        print(f'time used: {m:02.0f} mins {s:02.1f} secs.')

        # message for more visualization scripts
        msg = """Explore more info & visualization options with the following scripts:
        info.py                    # check HDF5 file structure and metadata
        view.py                    # 2D map view
        tsview.py                  # 1D point time-series (interactive)
        transect.py                # 1D profile (interactive)
        plot_coherence_matrix.py   # plot coherence matrix for one pixel (interactive)
        plot_network.py            # plot network configuration of the dataset
        plot_transection.py        # plot 1D profile along a line of a 2D matrix (interactive)
        save_kmz.py                # generate Google Earth KMZ file in raster image
        save_kmz_timeseries.py     # generate Google Earth KMZ file in points for time-series (interactive)
        """
        if print_aux:
            print(msg)


    def close(self, normal_end=True):
        # go back to original directory
        print('Go back to directory:', self.cwd)
        os.chdir(self.cwd)
        # message
        if normal_end:
            msg  = '\n################################################'
            msg += '\n   Normal end of smallbaselineApp processing!'
            msg += '\n################################################'
            print(msg)


def run_smallbaselineApp(inps):
    """Run the small baseline time series analsysis workflow."""
    start_time = time.time()

    # open and run
    app = TimeSeriesAnalysis(inps.customTemplateFile, inps.workDir)
    app.open()
    app.run(steps=inps.runSteps)

    # plot if:
    # a) --plot in command line OR
    # b) template['mintpy.plot'] = yes AND runSteps > 1
    if inps.plot or (app.template['mintpy.plot'] and len(inps.runSteps) > 1):
        app.plot_result()

    # close
    app.close()

    # used time
    m, s = divmod(time.time()-start_time, 60)
    print(f'Time used: {m:02.0f} mins {s:02.1f} secs\n')

    return
