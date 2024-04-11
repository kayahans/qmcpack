import testing
from testing import divert_nexus,restore_nexus,clear_all_sims
from testing import failed,FailedTest

pseudo_inputs = dict(
    pseudo_dir = 'pseudopotentials',
    pseudo_files_create = ['C.BFD.upf'],
    )

def get_system():
    from physical_system import generate_physical_system

    system = generate_physical_system(
        units  = 'A',
        axes   = [[ 1.785,  1.785,  0.   ],
                  [ 0.   ,  1.785,  1.785],
                  [ 1.785,  0.   ,  1.785]],
        elem   = ['C','C'],
        pos    = [[ 0.    ,  0.    ,  0.    ],
                  [ 0.8925,  0.8925,  0.8925]],
        tiling = (1,1,1),
        kgrid  = (1,1,1),
        kshift = (0,0,0),
        C      = 4
        )

    return system
#end def get_system


def get_pwscf_sim(type='scf'):
    from nexus_base import nexus_core
    from machines import job
    from pwscf import Pwscf,generate_pwscf

    nexus_core.runs = ''

    sim = None

    if type=='scf':
        sim = generate_pwscf(
            identifier   = 'scf',
            path         = 'scf',
            job          = job(machine='ws1',cores=1),
            input_type   = 'generic',
            calculation  = 'scf',
            input_dft    = 'lda', 
            ecutwfc      = 200,   
            nbnd         = 8,
            conv_thr     = 1e-8, 
            nosym        = True,
            wf_collect   = True,
            system       = get_system(),
            pseudos      = ['C.BFD.upf'], 
            nogamma      = True,
            )
    else:
        failed()
    #end if

    assert(sim is not None)
    assert(isinstance(sim,Pwscf))

    return sim
#end def get_pwscf_sim

def test_import():
    import pwscf_error_handler
#end def test_import
    
def test_bands_unconverged():
    import os
    import pwscf_error_handler
    tpath = testing.setup_unit_test_output_directory('pwscf_error_handler', 'test_bands_unconverged', **pseudo_inputs)
    sim = get_pwscf_sim('scf')

    assert(sim.locdir.rstrip('/')==os.path.join(tpath,'scf').rstrip('/'))
    if not os.path.exists(sim.locdir):
        os.makedirs(sim.locdir)
    #end if

    assert(not sim.finished)

    try:
        sim.check_sim_status()
    except IOError:
        None
    except Exception as e:
        failed(str(e))
    #end try

    assert(not sim.finished)

    out_path = os.path.join(tpath,'scf',sim.outfile)
    sim.write_inputs(save_image=False)

    out_text = ''
    outfile = open(out_path,'w')
    outfile.write(out_text)
    outfile.close()
    assert(os.path.exists(out_path))

    sim.check_sim_status()

    assert(not sim.finished)

    out_text = 'too many bands are not converged'
    outfile = open(out_path,'w')
    outfile.write(out_text)
    outfile.close()
    assert(out_text in open(out_path,'r').read())

    sim.check_sim_status()

    pwscf_error_handler.fix_bands(sim)
    sim.write_inputs(save_image=False)

    out_text = 'JOB DONE'
    outfile = open(out_path,'w')
    outfile.write(out_text)
    outfile.close()
    assert(out_text in open(out_path,'r').read())

    sim.check_sim_status()
    assert(not sim.failed)

    clear_all_sims()
    restore_nexus()
#end def test_bands_unconverged

