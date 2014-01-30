## script to call hd_playback/run_test.py : for use on pycloud
from hd_playback import run_test_cloud

class SimArgs:
    use_diff_length = True
    cloud_proc_func = "extract_red"
    cloud_proc_mod  = "hd_utils.cloud_proc_funcs"
    execution  = 0
    animation  = 1
    simulation = 1
    parallel   = 3
    downsample = 1
    prompt     = False
    show_neighbors = False
    log    = False
    fake_data_transform = (0,0,0,0,0,0)
    trajopt_init = "openrave_ik"
    interactive = False
    remove_table = False
    
    use_ar_init = True
    ar_demo_file = ""
    ar_run_file  = ""
    
    use_base = False
    not_allow_base = False
    early_stop_portion = 0.5
    no_traj_resample = False
    

    closest_rope_hack = False
    closest_rope_hack_thresh = 0.01
    
    pot_threshold = 15
    select = "clusters"
    friction = 30.0   
    max_steps_before_failure = 5
    tps_bend_cost_init  = 0.1
    tps_bend_cost_final = 0.001
    tps_n_iter          = 50
    cloud_downsample    = 0.02
    
    ndemos              = -1
    rope_scaling_factor = 1.0
    state_save_fname  = ''
    demo_type         = ''
    demo_data_h5_prefix = ''
    init_state_h5     = ''
    init_demo_name    = ''
    init_perturb_name = ''
    state_save_fname  = ''


max_steps_map = {'overhand'    : 5,
                 'square_knot' : 9,
                 'figure_eight': 6,
                 'cow_hitch'   : 6,
                 'clove_hitch' : 6,
                 'pile_hitch'  : 7,
                 'double_overhand':8,
                 'slip_hand'   : 5}

def run_sim_test(cmdline_params):
    SimArgs.ndemos              = cmdline_params[0]
    SimArgs.demo_type           = cmdline_params[1]
    SimArgs.demo_data_h5_prefix = cmdline_params[2]
    SimArgs.init_state_h5       = cmdline_params[3]
    SimArgs.init_demo_name      = cmdline_params[4]
    SimArgs.init_perturb_name   = cmdline_params[5]
    SimArgs.rope_scaling_factor = cmdline_params[6]
    SimArgs.state_save_fname    = cmdline_params[7]

    SimArgs.max_steps_before_failure = max_steps_map[SimArgs.demo_type]

    if SimArgs.ndemos==12:
        SimArgs.select="auto"
        

    return run_test_cloud.main(SimArgs)
