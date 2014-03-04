"""
Misc functions that are useful in the top-level task-execution scripts
"""


def request_int_in_range(too_high_val):
    while True:
        print "this pdb is here because without it the raw_input doesn't work. Just hit c and then enter the number for now"
        import pdb
        pdb.set_trace()
        try:
            choice_ind = int(raw_input())
        except ValueError: 
            print "invalid selection. try again"           
            continue
        if choice_ind <= too_high_val:
            return choice_ind
        

