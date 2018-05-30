import logging
from timeit import default_timer as timer


from keras_wrapper.extra.callbacks import \
    PrintPerformanceMetricOnEpochEndOrEachNUpdates as EvalPerformanceWrapper

import model_zoo
from utils.prepare_data import build_dataset





class Trainer:
    def __init__(self, params):
        """
        Trainer object constructor
        Will instanciate appropriate variables from param

        :param params: all params
        :type params:
        """
        self.logger = params['logger']
        self.reload = params['reload']
        self.metrics = params['metrics']
        self.shuffle = params['shuffle']
        self.verbose = params['verbose']
        self.n_epochs = params['n_epochs']
        self.eval_each = params['eval_each']
        self.batch_size = params['batch_size']
        self.eval_on_sets = params['eval_on_sets']
        self.epochs_for_save = params['epochs_for_save']
        self.eval_each_epochs = params['eval_each_epochs']
        self.parallel_loaders = params['parallel_loaders']
        self.data_augmentation = params['data_augmentation']
        self.sampling_save_mode = params['sampling_save_mode']
        self.homogeneous_batches = params['homogeneous_batches']
        self.start_eval_on_epoch = params['start_eval_on_epoch']
        self.outputs_ids_dataset = params['outputs_ids_dataset']

        # Preparing dict for training
        self.training_params = self.get_training_params(params)

        ## If reload option is used try first to look for an existing model
        # before creating one
        if self.reload:
            self.model = model_zoo.load_or_create_model(params)
        else:
            self.model = model_zoo.create_model(params)

        self.dataset = build_dataset(params)

        ########### Callbacks
        self.buildCallbacks(params, self.model, self.dataset)
        ###########

        ########### Training
        self.train()

        ###########


    def train(self):
        total_start_time = timer()

        logging.debug('Starting training!')


        self.model.trainNet(self.dataset, self.training_params)

        total_end_time = timer()
        time_difference = total_end_time - total_start_time
        logging.info('In total is {0:.2f}s = {1:.2f}m'.format(time_difference,
                                                              time_difference
                                                              / 60.0))

    def buildCallbacks(self, model, dataset):
        """
            Builds the selected set of callbacks run during the training of
            the model
        """

        callbacks = []

        #TODO reread this code
        if self.metrics:
            # Evaluate training
            extra_vars = {'n_parallel_loaders': self.parallel_loaders}
            for s in self.eval_on_sets:
                extra_vars[s] = dict()
                extra_vars[s]['references'] = dataset.extra_variables[s][
                    self.outputs_ids_datasetoutputs_ids_dataset[0]]
                if dataset.dic_classes.get(self.outputs_ids_dataset[0]):
                    extra_vars['n_classes'] = len(
                        dataset.dic_classes[self.outputs_ids_dataset[0]])

            if self.eval_each_epochs:
                callback_metric = \
                    EvalPerformanceWrapper(model,
                                           dataset,
                                           gt_id=self.outputs_ids_dataset[0],
                                           metric_name=self.metrics,
                                           set_name=self.eval_on_sets,
                                           batch_size=self.batch_size,
                                           each_n_epochs=self.eval_each,
                                           extra_vars=extra_vars,
                                           reload_epoch=self.reload,
                                           save_path=model.model_path,
                                           start_eval_on_epoch=self.start_eval_on_epoch,
                                           write_samples=True,
                                           write_type=self.sampling_save_mode,
                                           verbose=self.verbose)

            callbacks.append(callback_metric)

        self.callbacks = callbacks

    def get_training_params(self, params):
        training_params = {'n_epochs': self.max_epoch,
                           'batch_size': self.batch_size,
                           'homogeneous_batches': self.homogeneous_batches,
                           'shuffle': True,
                           'epochs_for_save': self.epochs_for_save,
                           'verbose': self.verbose,
                           'eval_on_sets': self.eval_on_sets_keras,
                           'n_parallel_loaders': self.parallel_loaders,
                           'extra_callbacks': self.callbacks,
                           'reload_epoch': self.reload,
                           'data_augmentation': self.data_augmentation}
        return training_params
