
import os

from error_handler import ErrorHandler
from pwscf import Pwscf
from generic import obj
from glob import glob

def use_restart(pw_sim):
    pass

def use_clean(pw_sim):
    pass

def fix_walltime(pw_sim):
    pass

def fix_charge(pw_sim):
    pass

def fix_bands(pw_sim):
    pass

error_dispatch = {
    'walltime' : fix_walltime,
    'wrong_charge': fix_charge,
    'unconverged_bands':fix_bands,
}

# Restart using previous job
pwscf_restartable_error_labels =  {
    #User limited
    'Maximum CPU time exceeded' :'walltime',
    'Program stopped by user request':'user_stop',
    
    #Ionic convergence
    'history already reset at previous step: stopping' : 'bfgs_history',
    
    #Electronic convergence
    'convergence NOT achieved after' :'electronic_convergence',    
}

# Restart from scratch
pwscf_unrestartable_error_labels = {
    # Electronic convergence
    'charge is wrong' :'wrong_charge',
    'Cholesky failed in invchol' : 'cholesky_decomposition',
    'problems computing cholesky' :'cholesky_decomposition',
    'not orthogonal operation':'symmetry_not_orthogonal',
    'too many bands are not converged':'unconverged_bands',
    'S matrix not positive definite': 's_matrix_not_positive_definite',
    'zhegvd failed': 'zhegvd_failed',
    '[Q, R] = qr(X, 0) failed': 'QR_failed',
    'eigenvectors failed to converge': 'unconverged_eigenvectors',
    'dexx is negative':'negative_dexx',
    'Error in routine broyden': 'broyden_failure',

    # Memory error
    'Not enough space allocated for radial FFT: try restarting with a larger cell_factor': 'not_enough_fft_space',

    # Revise job
    'some nodes have no k-points': 'npools_too_high',

    # Berry phase
    'probably because G_par is NOT a reciprocal lattice vector': 'gpar_error',
}

class PwscfErrorHandler(ErrorHandler):
    def __init__(self, pw_sim : Pwscf, maximum_tries: int=3):
        # if isinstance(pw_sim.input_type, PwscfInput):
        path         = pw_sim.locdir
        infile_name  = pw_sim.infile
        outfile_name = pw_sim.outfile  
        errfile_name = pw_sim.errfile
        sim          = pw_sim
        # else:
            # self.error('Incorrect input type: {}, error handler type: PWSCF'.format(pw_sim.input_type))
        #end if 
        self.path = path
        self.abspath = os.path.abspath(path)
        self.infile_name = infile_name
        self.outfile_name = outfile_name
        self.errfile_name = errfile_name
        self.sim          = sim
        self.max_tries    = maximum_tries

        self.info = obj()

        self.error_labels = pwscf_restartable_error_labels and pwscf_unrestartable_error_labels

        self.warning_labels = []
        self.failed_sims = []

    def save_attempt(self):
        folder_prefix = 'pwscf_attempt_*'
        folder_path = os.path.join(self.path, folder_prefix)
        num_attempt = len(glob(folder_path))
        if num_attempt < self.max_tries:
            self.sim.save_attempt(num_attempt)
        else:
            self.sim.failed = True

    def process_restartable_error(self, error_text=None):
        self.save_attempt()
        self.sim.input.control.restart_mode = 'restart'
        if error_text is not None:
            error_tags = [pwscf_restartable_error_labels[text] for text in error_text]
        else:
            error_tags = self.parse()
        for error in error_tags:
            fix_function = error_dispatch[error]
            fix_function(self.sim)
        self.sim.reset_indicators()

    def process_unrestartable_error(self, error_text=None):
        self.save_attempt()
        self.sim.input.control.restart_mode = 'from_scratch'
        if error_text is not None:
            error_tags = [pwscf_unrestartable_error_labels[text] for text in error_text]
        else:
            error_tags = self.parse()
        for error in error_tags:
            fix_function = error_dispatch[error]
            fix_function(self.sim)
        self.sim.reset_indicators()
        # Clean pwscf_output as well? 

    def parse(self):
        errors_found = self.parse_stdout() #+ self.parse_stderr
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
        errors_found = []
        try:
            lines = open(file,'r').read().splitlines()
            for l in lines:
                for error_str in error_strings:
                    if error_str in l:
                        errors_found.append(error_str)
            errors_found = list(set(errors_found))
        except:
            self.warn('file read failed')
        #end try 

        return errors_found

    