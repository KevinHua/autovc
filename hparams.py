# NOTE: If you want full control for model architecture. please take a look
# at the code and change whatever you want. Some hyper parameters are hardcoded.


class Map(dict):
	"""
    Example:
    m = Map({'first_name': 'Eduardo'}, last_name='Pool', age=24, sports=['Soccer'])

    Credits to epool:
    https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

	def __init__(self, *args, **kwargs):
		super(Map, self).__init__(*args, **kwargs)
		for arg in args:
			if isinstance(arg, dict):
				for k, v in arg.items():
					self[k] = v

		if kwargs:
			for k, v in kwargs.iteritems():
				self[k] = v

	def __getattr__(self, attr):
		return self.get(attr)

	def __setattr__(self, key, value):
		self.__setitem__(key, value)

	def __setitem__(self, key, value):
		super(Map, self).__setitem__(key, value)
		self.__dict__.update({key: value})

	def __delattr__(self, item):
		self.__delitem__(item)

	def __delitem__(self, key):
		super(Map, self).__delitem__(key)
		del self.__dict__[key]


# Default hyperparameters:
{
  "name": "wavenet_vocoder",
  "input_type": "raw",
  "quantize_channels": 65536,
  "preprocess": "preemphasis",
  "postprocess": "inv_preemphasis",
  "global_gain_scale": 0.55,
  "sample_rate": 16000,
  "silence_threshold": 2,
  "num_mels": 80,
  "fmin": 80,
  "fmax": 8000,
  "fft_size": 1024,
  "hop_size": 256,
  "frame_shift_ms": None,
  "win_length": 1024,
  "win_length_ms": -1.0,
  "window": "hann",
  "highpass_cutoff": 70.0,
  "output_distribution": "Logistic",
  "log_scale_min": -32.23619130191664,
  "out_channels": 30,
  "layers": 24,
  "stacks": 4,
  "residual_channels": 512,
  "gate_channels": 512,
  "skip_out_channels": 256,
  "dropout": 0.05,
  "kernel_size": 3,
  "cin_channels": 80,
  "cin_pad": 2,
  "upsample_conditional_features": True,
  "upsample_net": "ConvInUpsampleNetwork",
  "upsample_params": {
    "upsample_scales": [
      4,
      4,
      4,
      4
    ]
  },
  "gin_channels": -1,
  "n_speakers": -1,
  "pin_memory": True,
  "num_workers": 2,
  "batch_size": 3,
  "optimizer": "Adam",
  "optimizer_params": {
    "lr": 0.001,
    "eps": 1e-08,
    "weight_decay": 0.0
  },
  "lr_schedule": "step_learning_rate_decay",
  "lr_schedule_kwargs": {
    "anneal_rate": 0.5,
    "anneal_interval": 200000
  },
  "max_train_steps": 1000000,
  "nepochs": 2000,
  "clip_thresh": -1,
  "max_time_sec": None,
  "max_time_steps": 10240,
  "exponential_moving_average": True,
  "ema_decay": 0.9999,
  "checkpoint_interval": 100000,
  "train_eval_interval": 100000,
  "test_eval_epoch_interval": 50,
  "save_optimizer_state": True
}

hparams = Map({
	# Convenient model builder
	'builder': "wavenet",

   #
  "name": "wavenet_vocoder",
  "input_type": "raw",
  "quantize_channels": 65536,
  "preprocess": "preemphasis",
  "postprocess": "inv_preemphasis",
  "global_gain_scale": 0.55,
  "sample_rate": 16000,
  "silence_threshold": 2,
  "num_mels": 80,
  "fmin": 80,
  "fmax": 8000,
  "fft_size": 1024,
  "hop_size": 256,
  "frame_shift_ms": None,
  "win_length": 1024,
  "win_length_ms": -1.0,
  "window": "hann",
  "highpass_cutoff": 70.0,
  "output_distribution": "Logistic",
  "log_scale_min": -32.23619130191664,
  "out_channels": 30,
  "layers": 24,
  "stacks": 4,
  "residual_channels": 512,
  "gate_channels": 512,
  "skip_out_channels": 256,
  "dropout": 0.05,
  "kernel_size": 3,
  "cin_channels": 80,
  "cin_pad": 2,
  "upsample_conditional_features": True,
  "upsample_net": "ConvInUpsampleNetwork",
  "upsample_params": {
    "upsample_scales": [
      4,
      4,
      4,
      4
    ]
  },
  "gin_channels": -1,
  "n_speakers": -1,
  "pin_memory": True,
  "num_workers": 2,
  "batch_size": 3,
  "optimizer": "Adam",
  "optimizer_params": {
    "lr": 0.001,
    "eps": 1e-08,
    "weight_decay": 0.0
  },
  "lr_schedule": "step_learning_rate_decay",
  "lr_schedule_kwargs": {
    "anneal_rate": 0.5,
    "anneal_interval": 200000
  },
  "max_train_steps": 1000000,
  "nepochs": 2000,
  "clip_thresh": -1,
  "max_time_sec": None,
  "max_time_steps": 10240,
  "exponential_moving_average": True,
  "ema_decay": 0.9999,
  "checkpoint_interval": 100000,
  "train_eval_interval": 100000,
  "test_eval_epoch_interval": 50,
  "save_optimizer_state": True
})


def hparams_debug_string():
	values = hparams.values()
	hp = ['  %s: %s' % (name, values[name]) for name in sorted(values)]
	return 'Hyperparameters:\n' + '\n'.join(hp)
