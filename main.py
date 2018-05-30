import logging
import optparse

from constants import DEFAULT_CONFIG_FILE, DEFAULT_OUTPUT, DEFAULT_LOGGING
from utils.parameters import load_params, check_params
from trainer import Trainer
from chatbot import Chatbot

logging.basicConfig(level=logging.DEBUG,
                    filename=DEFAULT_LOGGING,
                    format='[%(asctime)s] %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S')

logger = logging.getLogger(__name__)


def parse_args():
    parser = optparse.OptionParser()
    parser.add_argument("-c", "--conf",
                        type=str, default=DEFAULT_CONFIG_FILE,
                        dest="cfg",
                        help="Source configuration file")
    parser.add_argument("-o", "--output",
                        type=str, default=DEFAULT_OUTPUT,
                        dest="output_dir",
                        help="Source configuration file")
    (opts, _) = parser.parse_args()
    return opts.cfg, opts.output_dir


if __name__ == "__main__":

    cfg, output_dir = parse_args()

    params = load_params(cfg, output_dir, logger)

    params = check_params(params, logger)

    if params['mode'] == 'training':
        logging.info('Running training.')
        trainer = Trainer(params)
        trainer.train()

    elif params['mode'] == 'testing':
        logging.info('Running testing.')
        chatbot = Chatbot(params)
        chatbot.run()
    else:
        logger.error('')
        loggi
