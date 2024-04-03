
import os

from error_handler import ErrorHandler
from simulation import Simulation
from pwscf import Pwscf
from pwscf_input import PwscfInput
from generic import obj
from fileio import TextFile
from copy import deepcopy
from glob import glob

def use_restart(pw_sim):
    pass

def use_clean(pw_sim):
    pass

def fix_walltime(pw_sim):
    pass

def fix_charge(pw_sim):
    pass

error_dispatch = {
    'walltime' : fix_walltime,
    'wrong_charge': fix_charge,
}

class PwscfErrorHandler(ErrorHandler):
    def __init__(self, pw_sim : Pwscf):
        if isinstance(pw_sim.input_type, PwscfInput):
            path = pw_sim.locdir
            infile_name = pw_sim.infile
            outfile_name = pw_sim.outfile  
            errfile_name = pw_sim.errfile
            sim          = pw_sim
        else:
            self.error('Incorrect input type: {}, error handler type: PWSCF'.format(pw_sim.input_type))
        #end if 
        self.path = path
        self.abspath = os.path.abspath(path)
        self.infile_name = infile_name
        self.outfile_name = outfile_name
        self.errfile_name = errfile_name
        self.sim          = sim

        self.info = obj()
        if self.infile_name is not None:
            self.input = PwscfInput(os.path.join(self.path,self.infile_name))
        #end if 

        self.error_labels = {
                'Maximum CPU time exceeded' :'walltime',
                'charge is wrong' :'wrong_charge',
                'convergence NOT achieved after' :'electronic_convergence',
                'history already reset at previous step: stopping' : 'bfgs_history',
                'Cholesky failed in invchol' : 'cholesky_decomposition',
                'problems computing cholesky' :'cholesky_decomposition',
                'not orthogonal operation':'symmetry_not_orthogonal',
                'dexx is negative':'negative_dexx',
                'too many bands are not converged':'unconverged_bands',
                'S matrix not positive definite': 's_matrix_not_positive_definite',
                'zhegvd failed': 'zhegvd_failed',
                '[Q, R] = qr(X, 0) failed': 'QR_failed',
                'probably because G_par is NOT a reciprocal lattice vector': 'gpar_error', # Berry phase
                'eigenvectors failed to converge': 'unconverged_eigenvectors',
                'Error in routine broyden': 'broyden_failure',
                'Not enough space allocated for radial FFT: try restarting with a larger cell_factor': 'not_enough_fft_space',
                'some nodes have no k-points': 'npools_too_high',
        }

        self.warning_labels = []
        self.failed_sims = []

    def save_attempt(self):
        folder_prefix = 'pwscf_attempt_*'
        folder_path = os.path.join(self.path, folder_prefix)
        num_attempt = len(glob(folder_path))

    def process_error(self):
        self.pw_sim.save_attempt()
        error_tags = self.parse()
        for error in error_tags:
            fix_function = error_dispatch[error]
            fix_function(self.pw_sim)
        self.pw_sim.reset_indicators()

    def parse(self):
        errors_found = self.parse_stdout + self.parse_stderr
        return errors_found
    
    def parse_stdout(self):
        outfile = os.path.join(self.path, self.outfile_name)
        return self.parse_error(outfile)

    def parse_stderr(self):
        errfile = os.path.join(self.path, self.errfile_name)
        return self.parse_error(errfile)

    def parse_error(self, file_name):
        file = os.path.join(self.path, file_name)
        error_strings = self.error_labels.keys()

        try:
            lines = open(file,'r').read().splitlines()
            errors_found = []
            for l in lines:
                for error_str in error_strings:
                    if error_str in l:
                        errors_found.append(error_str)
            errors_found = list(set(errors_found))
        except:
            self.warn('file read failed')
        #end try 

        return errors_found

    