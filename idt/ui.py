# /////////////////////// ADD BREAKPOINTS //////////////////////



# colorized text output
from termcolor import colored

import sys

# for programmatically generating Jupyter markdown from python strings
from IPython.display import display, Markdown, Latex



# debug UI switch
_debug_print_level_ = 1

# verbosity switch (e.g. 1: display data printers like [>], 2: display task indicators like [+])
_verbosity_level_ = 3

# debug code execution switch
_debug_exe_level_ = 1



# debug print function
def DBG(message, level):

    global _debug_level_

    if type(level) == int and level == _debug_print_level_:
        icon_buffer = colored("[" + str(level) + "]","light_grey","on_dark_grey", attrs=["bold", "blink"])
        message_buffer = colored(" " + message + " ","light_grey", attrs=["blink"])
        print("\n" + icon_buffer + message_buffer + "\n")

    else:
        return



# highlight print message
def HLT(message):
    icon_buffer = colored("[!]", "black", "on_yellow", attrs=["bold", "blink"])
    message_buffer = colored(" " +  message + " ", "yellow", attrs=["bold", "blink"])
    print("\n" + icon_buffer + message_buffer + "\n")

    return



# new task print function
def NEW(task_descriptor):  # add time profiler?

    if _verbosity_level_ > 0:
        icon_buffer = colored("[+]", "black", "on_blue", attrs=["bold"])
        task_descriptor_buffer = colored(task_descriptor, "blue", attrs=["bold"])
        print("\n" + icon_buffer + " " + task_descriptor_buffer)

    return



# new task print function
def PLT(task_descriptor):  # add time profiler?

    if _verbosity_level_ > 0:
        icon_buffer = colored("[+]", "black", "on_white", attrs=["bold"])
        task_descriptor_buffer = colored(task_descriptor, "white", attrs=["bold"])
        print("\n" + icon_buffer + " " + task_descriptor_buffer)

    return



# (next) update print function
def NXT(update_descriptor):  # add time profiler?

    if _verbosity_level_ > 1:
        icon_buffer = colored("[*]", "blue", attrs=["bold"])
        update_descriptor_buffer = colored(" " + update_descriptor, "light_blue")
        print("\n" + icon_buffer + update_descriptor_buffer)

    return



# end task print function
def END(success_message):  # add (end to) time profiler?

    if _verbosity_level_ > 1:
        icon_buffer = colored("[" + u'\u2713' + "]", "black", "on_light_green", attrs=["bold"])
        success_message_buffer = colored(" " + success_message, "light_green", attrs=["bold"])
        print("\n" + icon_buffer + success_message_buffer + "\n")

    return



# conditional print function
def CND(LHS_descriptor, RHS_descriptor, LHS, RHS, condition_descriptor, print_LHS=True, print_RHS=True):

    if _verbosity_level_ > 2:
        icon_buffer = colored("[{]", "black", "on_magenta")
        conditional_descriptor_buffer = colored(LHS_descriptor, "magenta") + " " + colored(condition_descriptor, "light_magenta", attrs=["bold"]) + " " + colored(RHS_descriptor, "magenta")
        LHS_buffer = colored("     " + LHS_descriptor + "  ::  " + str(LHS), "magenta")
        RHS_buffer = colored("     " + RHS_descriptor + "  ::  " + str(RHS), "magenta")
        print("\n" + icon_buffer + " " + conditional_descriptor_buffer)
        if print_LHS:
            print("\n" + LHS_buffer)
        if print_RHS:
            print("\n" + RHS_buffer)

    return



# error print function
def ERR(error_message):

    icon_buffer = colored("[E]", "black", "on_red", attrs=["bold", "blink"])
    error_message_buffer = colored(" " + error_message, "red", attrs=["bold"])
    print("\n" + icon_buffer + error_message_buffer + "\n")

    #sys.exit(1)



def MRK(markdown):
    display(Markdown(markdown))



def HLN():
    display(Markdown(r'''---'''))