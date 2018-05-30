
from constants import DEFAULT_CONFIG_FILE, DEFAULT_OUTPUT


def load_params(cfg):

    #TODO Find a better way of loading parameters but still keep the
    # possibility of loading params from a variable file (json?)
    default_params = __import__(DEFAULT_CONFIG_FILE).PARAMS
    if cfg != DEFAULT_CONFIG_FILE:
        cfg_params = __import__(cfg).PARAMS
        for k,v in cfg_params:
            default_params[k]=v

    return default_params


def check_params(params,logger):
    #TODO
    #must change logging.basicConfig

    try:



        # only two modes are allowed
        assert (params["mode"] == "training" or params["mode"]=="testing")

    except AssertionError as e:
        logger.error(e)
        raise e

    return params
