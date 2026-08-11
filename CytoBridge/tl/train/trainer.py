import torch
import copy
import json
import numpy as np
import pandas as pd
import anndata as ad
from tqdm import tqdm
from CytoBridge.tl.core.methods import neural_ode_step, ODEFunc, ODEFunc2InteractionEnergy
from CytoBridge.utils.utils import sample,trace_df_dz,compute_integral, set_seed
from CytoBridge.tl.core.losses import calc_ot_loss, calc_mass_loss, calc_score_matching_loss,Density_loss,calc_pinn_loss
from CytoBridge.tl.core import methods
from CytoBridge.tl.core.models import DynamicalModel
from CytoBridge.tl.core.flow_matching import SchrodingerBridgeConditionalFlowMatcher, ConditionalRegularizedUnbalancedFlowMatcher, get_batch_size, compute_uot_plans, get_batch_uot_fm
from CytoBridge.tl.downstream.analysis import simulate_trajectory
import math
import os
import platform
import sys
import time as wallclock
import ot
from torchdiffeq import odeint
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau  # Import StepLR scheduler
class TrainingPipeline:
    def __init__(
        self,
        model,
        config,
        batch_size,
        device,
        data,
        *,
        seed_already_applied: bool = False,
        run_context: dict | None = None,
    ):  # Added 'data' parameter for initialization
        self.model = model
        self.config = config
        self.batch_size = batch_size
        self.optimizer = None
        self.scheduler = None  # Initialize scheduler variable
        self.device = device
        seed = self.config.get('seed')
        if seed is not None and not seed_already_applied:
            set_seed(seed)
        self.model.to(device)
        # Determine if mass component is used based on model configuration
        self.use_mass = 'growth' in self.config['model']['components']
        # Determine if score component is used based on model configuration
        self.use_score = 'score' in self.config['model']['components']
        # Determine if interaction component is used based on model configuration
        self.use_interaction = 'interaction' in self.config['model']['components']

        # Initialize ODE function (unified gradient calculation entry)
        self.ode_func = ODEFunc(
            model=self.model,
            sigma=config['training']['defaults'].get('sigma', 0.05),
            use_mass=self.use_mass,
            score_use=self.use_score,
            interaction_use=self.use_interaction
        )

        # New: Initialize variables required for train_score_model
        self.logger = self._setup_logger()  # Simple logger implementation
        # Get experiment directory from configuration (default to './results' if not specified)
        self.exp_dir = self.config.get('ckpt_dir', './results')
        os.makedirs(self.exp_dir, exist_ok=True)
        self.training_history = []
        self._active_stage_index = None
        self._optimizer_step_count = 0
        self._stage_summaries = []
        self._training_run_summary = {}
        self._run_context = dict(run_context or {})
        self._initial_batch_size = int(batch_size)
        self._model_parameter_count = int(
            sum(parameter.numel() for parameter in self.model.parameters())
        )
        self._model_trainable_parameter_count_at_start = int(
            sum(
                parameter.numel()
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
        )
        self._data_sample_counts = [int(value.shape[0]) for value in data]
        self._input_dimension = (
            int(data[0].shape[1]) if data and data[0].ndim >= 2 else None
        )
        # Construct DataFrame from input data to fit the format required by train_score_model
        self.df = self._prepare_df(data)
        # Get sorted list of unique time points (grouped by 'samples' column)
        self.groups = sorted(self.df.samples.unique())

    def _setup_logger(self):
        """Simple logger implementation to replace the original logger"""

        class SimpleLogger:
            @staticmethod
            def info(msg):
                print(f"[INFO] {msg}")

        return SimpleLogger()

    @staticmethod
    def _history_float(value):
        if value is None:
            return float('nan')
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().item()
        return float(value)

    def _current_learning_rate(self):
        optimizer = getattr(self, 'optimizer', None)
        if optimizer is None or not getattr(optimizer, 'param_groups', None):
            return float('nan')
        return float(optimizer.param_groups[0]['lr'])

    @staticmethod
    def _nullable_float(value):
        """Return a finite JSON float, or ``None`` when unavailable."""
        try:
            converted = float(value)
        except (TypeError, ValueError):
            return None
        return converted if math.isfinite(converted) else None

    @staticmethod
    def _cpu_max_rss_mib():
        """Return the process-lifetime RSS high-water mark when supported."""
        try:
            import resource

            rss = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        except (ImportError, AttributeError, OSError, TypeError, ValueError):
            return None
        # macOS reports bytes; Linux and the BSDs report KiB.
        divisor = 1024.0**2 if sys.platform == 'darwin' else 1024.0
        value = rss / divisor
        return value if math.isfinite(value) else None

    def _cuda_device(self):
        device = getattr(self, 'device', None)
        try:
            parsed = device if isinstance(device, torch.device) else torch.device(device)
        except (TypeError, ValueError, RuntimeError):
            return None
        if parsed.type != 'cuda' or not torch.cuda.is_available():
            return None
        return parsed

    def _reset_cuda_peak_memory(self):
        device = self._cuda_device()
        if device is None:
            return
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except (RuntimeError, AssertionError):
            return

    def _synchronize_device(self):
        """Synchronize CUDA so wall-clock intervals include queued kernels."""
        device = self._cuda_device()
        if device is None:
            return
        try:
            torch.cuda.synchronize(device)
        except (RuntimeError, AssertionError):
            return

    def _wall_time_start(self):
        self._synchronize_device()
        return wallclock.perf_counter()

    def _wall_time_elapsed(self, started_at):
        self._synchronize_device()
        return max(0.0, wallclock.perf_counter() - started_at)

    def _cuda_peak_memory_mib(self):
        device = self._cuda_device()
        if device is None:
            return {"allocated": None, "reserved": None}
        try:
            divisor = 1024.0**2
            return {
                "allocated": float(torch.cuda.max_memory_allocated(device)) / divisor,
                "reserved": float(torch.cuda.max_memory_reserved(device)) / divisor,
            }
        except (RuntimeError, AssertionError):
            return {"allocated": None, "reserved": None}

    def _optimizer_step(self):
        """Perform and count one optimizer update owned by this pipeline."""
        self.optimizer.step()
        if hasattr(self, '_optimizer_step_count'):
            self._optimizer_step_count += 1

    def _mark_selected_checkpoint(self, selected_epoch):
        """Separate the selected checkpoint from record-setting ``is_best`` rows."""
        stage_index = int(
            getattr(self, '_active_stage_index', 0)
            if getattr(self, '_active_stage_index', None) is not None
            else 0
        )
        for row in getattr(self, 'training_history', []):
            if int(row.get('stage_index', -1)) != stage_index:
                continue
            row['is_selected_checkpoint'] = (
                selected_epoch is not None and int(row['epoch']) == int(selected_epoch)
            )

    def _finalize_active_stage(
        self,
        *,
        stage_params,
        stage_started_at,
        learning_rate_start,
        optimizer_steps_start,
    ):
        """Attach measured stage totals to history and the run-level summary."""
        stage_index = int(self._active_stage_index)
        stage_wall_time = self._wall_time_elapsed(stage_started_at)
        learning_rate_end = self._nullable_float(self._current_learning_rate())
        optimizer_steps = (
            int(self._optimizer_step_count - optimizer_steps_start)
            if hasattr(self, '_optimizer_step_count')
            else None
        )
        cuda_peak = self._cuda_peak_memory_mib()
        stage_rows = [
            row
            for row in self.training_history
            if int(row.get('stage_index', -1)) == stage_index
        ]
        for row in stage_rows:
            row['stage_wall_time_seconds'] = float(stage_wall_time)
            row['stage_learning_rate_start'] = learning_rate_start
            row['stage_learning_rate_end'] = learning_rate_end
            row['stage_optimizer_steps'] = optimizer_steps

        selected = next(
            (
                int(row['epoch'])
                for row in stage_rows
                if bool(row.get('is_selected_checkpoint', False))
            ),
            None,
        )
        self._stage_summaries.append(
            {
                'stage_index': stage_index,
                'stage': str(stage_params['name']),
                'mode': str(stage_params.get('mode', 'neural_ode')),
                'configured_epochs': int(stage_params['epochs']),
                'recorded_epochs': len(stage_rows),
                'batch_size': int(stage_params.get('batch_size', self.batch_size)),
                'learning_rate_start': learning_rate_start,
                'learning_rate_end': learning_rate_end,
                'optimizer_step_count': optimizer_steps,
                'wall_time_seconds': float(stage_wall_time),
                'cuda_peak_allocated_mib': cuda_peak['allocated'],
                'cuda_peak_reserved_mib': cuda_peak['reserved'],
                'trainable_parameter_count': int(
                    sum(
                        parameter.numel()
                        for parameter in self.model.parameters()
                        if parameter.requires_grad
                    )
                ),
                'save_strategy': str(stage_params.get('save_strategy', 'best')),
                'selected_checkpoint_epoch': selected,
            }
        )

    def _build_training_run_summary(self, run_wall_time_seconds):
        stage_allocated = [
            value
            for value in (
                stage['cuda_peak_allocated_mib'] for stage in self._stage_summaries
            )
            if value is not None
        ]
        stage_reserved = [
            value
            for value in (
                stage['cuda_peak_reserved_mib'] for stage in self._stage_summaries
            )
            if value is not None
        ]
        cuda_device = self._cuda_device()
        try:
            cuda_device_name = (
                torch.cuda.get_device_name(cuda_device)
                if cuda_device is not None
                else None
            )
        except (RuntimeError, AssertionError):
            cuda_device_name = None
        try:
            cuda_device_index = (
                int(cuda_device.index)
                if cuda_device is not None and cuda_device.index is not None
                else (
                    int(torch.cuda.current_device())
                    if cuda_device is not None
                    else None
                )
            )
        except (RuntimeError, AssertionError):
            cuda_device_index = None
        try:
            cudnn_version = torch.backends.cudnn.version()
        except (AttributeError, RuntimeError):
            cudnn_version = None
        context = dict(self._run_context)
        context.setdefault('model_input_dim', self._input_dimension)
        context.setdefault('n_observations', int(sum(self._data_sample_counts)))
        context.setdefault('n_timepoints', len(self._data_sample_counts))
        context.setdefault('sample_counts_by_timepoint', self._data_sample_counts)
        batch_sizes = [int(stage['batch_size']) for stage in self._stage_summaries]
        return {
            'schema_version': 1,
            'scope': (
                'TrainingPipeline.train only; excludes post-training inference, '
                'evaluation, and AnnData serialization.'
            ),
            'timing_scope': (
                'Stage time includes optimizer setup, stage-specific preparation, '
                'epochs, checkpoint selection, and checkpoint write. Epoch time '
                'covers one training iteration through its optimizer update. CUDA '
                'is synchronized at timing boundaries. Run time ends after the '
                'final history write and excludes summary serialization.'
            ),
            'memory_scope': (
                'CPU max RSS is the process-lifetime high-water mark sampled after '
                'training. CUDA peaks are per-stage allocator high-water marks; the '
                'run value is the maximum across stages.'
            ),
            'timing': {
                'run_wall_time_seconds': float(run_wall_time_seconds),
                'stage_wall_time_seconds_sum': float(
                    sum(stage['wall_time_seconds'] for stage in self._stage_summaries)
                ),
                'epoch_wall_time_seconds_sum': float(
                    sum(
                        self._nullable_float(row.get('epoch_wall_time_seconds')) or 0.0
                        for row in self.training_history
                    )
                ),
            },
            'resources': {
                'cpu_max_rss_mib': self._cpu_max_rss_mib(),
                'cuda_peak_allocated_mib': (
                    float(max(stage_allocated)) if stage_allocated else None
                ),
                'cuda_peak_reserved_mib': (
                    float(max(stage_reserved)) if stage_reserved else None
                ),
            },
            'environment': {
                'device': str(getattr(self, 'device', 'unknown')),
                'device_type': (
                    str(getattr(self.device, 'type', self.device))
                    if hasattr(self, 'device')
                    else None
                ),
                'torch_version': str(torch.__version__),
                'cuda_compiled_version': (
                    str(torch.version.cuda) if torch.version.cuda is not None else None
                ),
                'cuda_available': bool(torch.cuda.is_available()),
                'cuda_device_name': cuda_device_name,
                'cuda_device_index': cuda_device_index,
                'cudnn_version': (
                    int(cudnn_version) if cudnn_version is not None else None
                ),
                'python_version': platform.python_version(),
                'platform': platform.platform(),
            },
            'model': {
                'parameter_count': int(self._model_parameter_count),
                'trainable_parameter_count_at_start': int(
                    self._model_trainable_parameter_count_at_start
                ),
            },
            'data': context,
            'training': {
                'initial_batch_size': int(self._initial_batch_size),
                'stage_batch_sizes': batch_sizes,
                'optimizer_step_count': int(self._optimizer_step_count),
                'optimizer_step_count_scope': (
                    'Successful optimizer.step calls executed by TrainingPipeline.'
                ),
            },
            'stages': list(self._stage_summaries),
        }

    def training_run_summary(self):
        """Return the measured training-only run summary."""
        return copy.deepcopy(getattr(self, '_training_run_summary', {}))

    def _save_training_run_summary(self):
        summary = self.training_run_summary()
        if not summary:
            return None
        output_dir = self.config.get('ckpt_dir', getattr(self, 'exp_dir', './results'))
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'training_run_summary.json')
        temporary_path = path + '.tmp'
        with open(temporary_path, 'w', encoding='utf-8') as handle:
            json.dump(summary, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write('\n')
        os.replace(temporary_path, path)
        return path

    def _record_training_epoch(
        self,
        *,
        stage_params,
        epoch,
        loss,
        checkpoint_metric,
        checkpoint_value,
        is_best,
        epoch_wall_time_seconds=None,
        optimizer_steps_epoch=None,
        **metrics,
    ):
        """Append one schema-stable, serializable row to the training history."""
        history = getattr(self, 'training_history', None)
        if history is None:
            self.training_history = []
            history = self.training_history
        effective_batch_size = stage_params.get(
            'batch_size', getattr(self, 'batch_size', None)
        )
        row = {
            'stage_index': int(
                getattr(self, '_active_stage_index', 0)
                if getattr(self, '_active_stage_index', None) is not None
                else 0
            ),
            'stage': str(stage_params['name']),
            'mode': str(stage_params.get('mode', 'neural_ode')),
            'epoch': int(epoch),
            'epochs': int(stage_params['epochs']),
            'loss': self._history_float(loss),
            'checkpoint_metric': str(checkpoint_metric),
            'checkpoint_value': self._history_float(checkpoint_value),
            'is_best': bool(is_best),
            'is_selected_checkpoint': False,
            'learning_rate': self._current_learning_rate(),
            'save_strategy': str(stage_params.get('save_strategy', 'best')),
            'batch_size': (
                int(effective_batch_size)
                if effective_batch_size is not None
                else float('nan')
            ),
            'epoch_wall_time_seconds': self._history_float(epoch_wall_time_seconds),
            'optimizer_steps_epoch': self._history_float(optimizer_steps_epoch),
            'optimizer_steps_cumulative': self._history_float(
                getattr(self, '_optimizer_step_count', None)
            ),
            'stage_wall_time_seconds': float('nan'),
            'stage_learning_rate_start': float('nan'),
            'stage_learning_rate_end': float('nan'),
            'stage_optimizer_steps': float('nan'),
        }
        for key, value in metrics.items():
            row[str(key)] = self._history_float(value)
        history.append(row)

        flush_every = int(
            self.config.get('training', {}).get('history_flush_every', 25)
        )
        if flush_every > 0 and len(history) % flush_every == 0:
            self._save_training_history()

    def training_history_frame(self):
        """Return the complete per-epoch history in training-plan order."""
        return pd.DataFrame(getattr(self, 'training_history', []))

    def _save_training_history(self):
        """Atomically persist the accumulated per-epoch history as CSV."""
        frame = self.training_history_frame()
        if frame.empty:
            return None
        output_dir = self.config.get('ckpt_dir', getattr(self, 'exp_dir', './results'))
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'training_history.csv')
        temporary_path = path + '.tmp'
        frame.to_csv(temporary_path, index=False)
        os.replace(temporary_path, path)
        return path

    def _prepare_df(self, data):
        """Construct DataFrame from input data to fit the format required by train_score_model
        
        Args:
            data: List of tensors where each element represents samples at a specific time point (shape: n_samples×2)
        
        Returns:
            pd.DataFrame: Combined DataFrame with columns 'x1', 'x2', and 'samples' (time point)
        """
        all_samples = []
        for t_idx, x in enumerate(data):
            x_np = x.cpu().detach().numpy()  # Convert tensor to numpy array
            # Construct DataFrame for current time point: columns = [x1, x2, samples (time point)]
            # Construct DataFrame for current time point
            data_dict = {'samples': np.full(x_np.shape[0], t_idx, dtype=np.float64)}
            for i in range(x_np.shape[1]):
                data_dict[f'x{i+1}'] = x_np[:, i]
            df_t = pd.DataFrame(data_dict)
            all_samples.append(df_t)
        # Concatenate DataFrames from all time points and reset index
        return pd.concat(all_samples, ignore_index=True)

    # --------------------------
    # Main Modifications: Optimizer and Scheduler Setup
    # --------------------------
    def _setup_stage(self, stage_params):
        lr = stage_params['lr']
        print(f"\n====  {stage_params['name']}  ====")

        # Get flags for score network training from stage parameters
        train_strategy = str(stage_params.get('train_strategy', '')).lower()

        if 'batch_size' in stage_params:
            self.batch_size = int(stage_params['batch_size'])



        if not train_strategy or train_strategy == 'none':
            use_v = train_g = use_s = use_i = True          # 缺省策略：全训练
        else:
            use_v, train_g, use_s, use_i = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy, 'i' in train_strategy

        score_use = stage_params.get('score_use', None)
        interaction_use = stage_params.get('interaction_use', None)
        if score_use is None:
            score_use = use_s
        if interaction_use is None:
            interaction_use = use_i

        if stage_params.get('mode') ==  "neural_ode":
            train_s = use_s
            self.model.use_growth_in_ode_inter = stage_params.get('use_growth_in_ode_inter', True)
            self.ode_func.use_mass = train_g  # Fixed: Use train_g from train_strategy, not use_growth_in_ode_inter
            self.ode_func.score_use = score_use
            self.ode_func.interaction_use = interaction_use
            print(f"  [DEBUG] ODEFunc flags: use_mass={self.ode_func.use_mass}, score_use={self.ode_func.score_use}, interaction_use={self.ode_func.interaction_use}")

        elif stage_params.get('mode') ==  "flow_matching":
            train_s = True
        elif stage_params.get('mode') == "score_matching":
            use_v = train_g = use_i = False
            train_s = True
        else:
            raise ValueError(f"Unknown training mode: {stage_params['mode']}")

        # Collect trainable parameters based on component flags
        params = []

        for name, module in self.model.named_children():
            # print(f"Name: {name}")
            # print(f"Module: {module}")
            # print(f"Module type: {type(module)}")
            # print("Parameters:")
            if (name == 'velocity_net' and use_v) or (name == 'growth_net' and train_g) or  (name == 'score_net' and train_s) or (name == 'interaction_net'  and use_i):
                if name == "interaction_net" and hasattr(module, "link_predictor"):
                    for p_name, p in module.named_parameters():
                        # Keep pretrained link predictor frozen in all stages.
                        if p_name.startswith("link_predictor."):
                            p.requires_grad = False
                        else:
                            p.requires_grad = True
                            params.append(p)
                    module.link_predictor.eval()
                else:
                    for p in module.parameters():
                        p.requires_grad = True
                        params.append(p)
                        # print(f"  Parameter shape: {p.shape}")
                        # print(f"  Parameter requires_grad: {p.requires_grad}")
                print("-" * 50)
            else:

                for p in module.parameters():
                    p.requires_grad = False
                    # print(f"  Parameter shape: {p.shape}")
                    # print(f"  Parameter requires_grad: {p.requires_grad}")
        # Initialize optimizer with only trainable parameters
        optimizer_type = str(stage_params.get('optimizer_type', 'adam')).lower()
        if optimizer_type == 'adamw':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, params), lr=lr
            )
        else:
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, params), lr=lr
            )

        # Reset scheduler before setting up new one
        self.scheduler = None
        if 'scheduler_type' in stage_params:
            if stage_params['scheduler_type'] == 'cosine':
                # Use Cosine Annealing scheduler if specified
                cosine_epochs = stage_params.get('cosine_epochs', 1000)
                cosine_eta_min = float(stage_params.get('scheduler_eta_min', 1e-7))
                self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, 
                    T_max=cosine_epochs, 
                    eta_min=cosine_eta_min
                )
            elif stage_params['scheduler_type'] == 'steplr':
                # Use StepLR scheduler if specified
                self.scheduler = StepLR(
                    optimizer=self.optimizer,
                    step_size=stage_params['scheduler_step_size'],
                    gamma=stage_params['scheduler_gamma']  # Learning rate decay factor
                )
                print(f"  Enabled learning rate scheduler: step_size={stage_params['scheduler_step_size']}, gamma={stage_params['scheduler_gamma']}")
            elif stage_params['scheduler_type'] == 'plateau':
                self.scheduler = ReduceLROnPlateau(
                    optimizer=self.optimizer,
                    mode='min',
                    factor=0.5,
                    patience=50,
                )
        else:
            print("  No scheduler parameters configured - keeping learning rate constant")

        # Print gradient status (trainable/non-trainable) for each module
        for n, m in self.model.named_children():
            flag = any(p.requires_grad for p in m.parameters())
            print(f"  {n:<15}  grad={flag}")
        # Print shapes of parameters in optimizer
        print("  Optimizer parameters (shapes):", [p.shape for g in self.optimizer.param_groups for p in g['params']])

    def train(self, data, time_points):
        """Main training loop that executes multiple training stages based on configuration
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            DynamicalModel: Trained model
        """
        run_started_at = self._wall_time_start()
        # Get training plan and base default parameters from configuration
        training_plan = self.config['training']['plan']
        base_defaults = self.config['training']['defaults']

        # Execute each stage in the training plan
        for stage_index, stage_config in enumerate(training_plan):
            self._active_stage_index = int(stage_index)
            stage_started_at = self._wall_time_start()
            optimizer_steps_start = int(self._optimizer_step_count)
            self._reset_cuda_peak_memory()
            # Merge base defaults with stage-specific config (stage config takes priority)
            stage_params = base_defaults.copy()
            stage_params.update(stage_config)
            stage_name = stage_params['name']


            print(f"\n--- Starting Stage: {stage_name} ---")
            print(
                f"  Mode: {stage_params['mode']}, Epochs: {stage_params['epochs']}, Use Score: {stage_params.get('score_use', False)}")
            train_strategy = stage_params.get('train_strategy', None)


            # Setup optimizer, scheduler, and trainable parameters for current stage
            self._setup_stage(stage_params)
            learning_rate_start = self._nullable_float(
                self._current_learning_rate()
            )

            # Execute stage training based on mode
            if stage_params['mode'] == 'neural_ode':
                self.run_neural_ode_stage(stage_params, data, time_points)
            elif stage_params['mode'] == 'flow_matching':
                self.run_flow_matching_stage(stage_params, data, time_points)
            elif stage_params['mode'] == 'score_matching':
                self.run_score_matching_stage(stage_params, data, time_points)
            else:
                raise ValueError(f"Unknown training mode: {stage_params['mode']}")
            self._finalize_active_stage(
                stage_params=stage_params,
                stage_started_at=stage_started_at,
                learning_rate_start=learning_rate_start,
                optimizer_steps_start=optimizer_steps_start,
            )
            self._save_training_history()

        self._active_stage_index = None
        self._save_training_history()
        self._training_run_summary = self._build_training_run_summary(
            self._wall_time_elapsed(run_started_at)
        )
        self._save_training_run_summary()
        return self.model

    def run_neural_ode_stage(self, stage_params, data, time_points):
        """Execute training stage using Neural ODE mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        epochs = stage_params['epochs']
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')


        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state = copy.deepcopy(self.model.state_dict())
        train_strategy = stage_params.get('train_strategy', None)
        train_name=stage_params["name"]

        # Training loop over epochs
        # Run training loop for the specified number of epochs
        checkpoint_metric = stage_params.get('checkpoint_metric', 'average_loss')
        best_epoch = None
        for epoch in range(1, epochs + 1):
            epoch_started_at = self._wall_time_start()
            optimizer_steps_before_epoch = getattr(
                self, '_optimizer_step_count', None
            )
            loss = self.train_neural_ode_epoch(stage_params, data, time_points, self.ode_func)

            # Print progress every 10 epochs
            if epoch % 10 == 0:
                print(f"  Stage '{stage_params['name']}', Epoch {epoch}/{epochs}, Loss: {loss:.4f}")
            # if epoch % 10 == 0 and self.use_interaction:
            #     plot_interaction_potential_epoch(self.model,d=1,num_points=40,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #     if epoch < 15:
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")

            # if "i" in train_strategy:
            #     if epoch % 10 == 0:
            #         plot_interaction_potential_epoch(self.model,d=1,num_points=21,output_path=self.config["ckpt_dir"]+f"/interfigures/{train_name}_epoch_{epoch}_inter",device="cuda")
            #         print(f"{train_name} plot_interaction_potential_epoch {epoch} has done")
            # Update best model if current loss is lower than previous best
            if checkpoint_metric == 'legacy_forward_last_ot':
                candidate_loss = self._last_neural_ode_epoch['forward_last_ot']
                candidate_state = self._last_neural_ode_epoch['state_after_forward']
            elif checkpoint_metric == 'average_loss':
                candidate_loss = loss
                candidate_state = self.model.state_dict()
            else:
                raise ValueError(
                    "checkpoint_metric must be 'average_loss' or "
                    "'legacy_forward_last_ot'."
                )
            candidate_loss = self._history_float(candidate_loss)
            is_best = candidate_loss < best_loss
            if is_best:
                best_loss = candidate_loss
                best_epoch = int(epoch)
                self.logger.info(f"Epoch {epoch:3d} has a lower loss| all_loss {best_loss:.4f}")
                best_state = copy.deepcopy(candidate_state)
            epoch_metrics = getattr(self, '_last_neural_ode_epoch', {})
            self._record_training_epoch(
                stage_params=stage_params,
                epoch=epoch,
                loss=loss,
                checkpoint_metric=checkpoint_metric,
                checkpoint_value=candidate_loss,
                is_best=is_best,
                epoch_wall_time_seconds=self._wall_time_elapsed(epoch_started_at),
                optimizer_steps_epoch=(
                    int(self._optimizer_step_count - optimizer_steps_before_epoch)
                    if optimizer_steps_before_epoch is not None
                    else None
                ),
                forward_last_ot=epoch_metrics.get('forward_last_ot'),
                mean_ot_loss=epoch_metrics.get('mean_ot_loss'),
                mean_mass_loss=epoch_metrics.get('mean_mass_loss'),
                mean_energy_loss=epoch_metrics.get('mean_energy_loss'),
                mean_density_loss=epoch_metrics.get('mean_density_loss'),
                mean_pinn_loss=epoch_metrics.get('mean_pinn_loss'),
                n_intervals=epoch_metrics.get('n_intervals'),
            )
            if self.scheduler is not None and not stage_params.get('scheduler_step_before_reverse', False):
                scheduler_metric = stage_params.get('scheduler_metric', 'average_loss')
                scheduler_value = (
                    self._last_neural_ode_epoch['forward_last_ot']
                    if scheduler_metric == 'forward_last_ot'
                    else loss
                )
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(scheduler_value)
                else:
                    self.scheduler.step()

        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state
            save_loss = best_loss
            selected_epoch = best_epoch
        else:  # 'last' strategy
            # Snapshot the state produced by the declared number of epochs.
            # Calling train_neural_ode_epoch here used to perform an undocumented
            # extra optimizer update (e.g. 1001 updates for a 1000-epoch stage).
            save_state = copy.deepcopy(self.model.state_dict())
            save_loss = loss
            selected_epoch = int(epoch)

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        save_path = os.path.join(ckpt_dir, ckpt_filename)
        torch.save(save_state, save_path)
        self._mark_selected_checkpoint(selected_epoch)
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) saved → {save_path}")

    def train_neural_ode_epoch(self, stage_params, data, time_points, ode_func):
        """Calculate loss for one epoch of Neural ODE training
        
        Args:
            stage_params: Dictionary of parameters for current stage (loss weights, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
            ode_func: ODEFunc instance for computing ODE updates
        
        Returns:
            float: Average loss over all time intervals
        """
        # Get loss weights and configuration from stage parameters
        lambda_ot = stage_params['lambda_ot']
        lambda_mass = stage_params['lambda_mass']
        lambda_energy = stage_params['lambda_energy']
        
        OT_loss_type = stage_params['OT_loss']
        alpha_spatial = stage_params.get('alpha_spatial', 1.0)
        alpha_express = stage_params.get('alpha_express', 1.0)
        use_density_loss = stage_params.get('use_density_loss', False)
        use_pinn_loss = stage_params.get('use_pinn_loss', False)

        global_mass = stage_params.get('global_mass', False)
        if use_density_loss:
            if 'density_top_k' not in stage_params or 'lambda_density' not in stage_params or 'density_hinge_value' not in stage_params:
                raise ValueError(
                    "When use_density_loss=True, all 'density_top_k','lambda_density' and 'density_hinge_value' "
                    "must be provided in stage_params.(Default recommended ( 5 , 10 and  0.01))" 
                )            
            top_k = stage_params['density_top_k']
            hinge_value = stage_params['density_hinge_value']
            lambda_density = stage_params['lambda_density']
            density_fn = Density_loss(hinge_value)


        # Initialize with sampled data from the first time point
        x0 = sample(data[0], self.batch_size).to(self.device)
        # Initialize log-weights (uniform distribution)
        lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
        # Total number of samples at the first time point
        mass_0 = data[0].shape[0]
        forward_mass0 = mass_0

        total_loss = 0.0
        valid_intervals = 0
        component_sums = {
            'ot_loss': 0.0,
            'mass_loss': 0.0,
            'energy_loss': 0.0,
            'density_loss': 0.0,
            'pinn_loss': 0.0,
        }
        expected_intervals = (len(time_points) - 1) * (
            2 if self.config.get('reverse', False) else 1
        )
        max_grad_norm = stage_params.get('max_grad_norm')
        optimizer_parameters = [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group['params']
        ]
        # Iterate over all time intervals (from t_{i-1} to t_i)
        for idx in range(1, len(time_points)):
            # Reset gradients before each time interval update
            self.optimizer.zero_grad()

            # Get current time interval and target data
            t0, t1 = time_points[idx - 1], time_points[idx]
            data_t1 = sample(data[idx], self.batch_size).to(self.device)
            # Total number of samples at the target time point
            mass_1 = data[idx].shape[0]
            # Calculate relative mass ratio between target and initial time points
            relative_mass = mass_1 / mass_0

            # Perform one Neural ODE step to predict state at t1
            x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

            if not torch.isfinite(x1).all() or not torch.isfinite(lnw1).all():
                raise FloatingPointError(
                    f"Non-finite ODE state during {stage_params['name']} "
                    f"forward interval {t0}->{t1}."
                )
            # Calculate individual loss components
            try:
                loss_ot = calc_ot_loss(
                    x1,
                    data_t1,
                    lnw1,
                    OT_loss_type,
                    alpha_spatial=alpha_spatial,
                    alpha_express=alpha_express,
                    spatial_dim=self.config.get('spatial_dim', 2),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"OT loss failed during {stage_params['name']} "
                    f"forward interval {t0}->{t1}."
                ) from exc
            if not torch.isfinite(loss_ot):
                raise FloatingPointError(
                    f"Non-finite OT loss during {stage_params['name']} "
                    f"forward interval {t0}->{t1}."
                )
            # Calculate mass loss only if mass component is enabled
            loss_mass = (
                calc_mass_loss(
                    x1,
                    data_t1,
                    lnw1,
                    relative_mass,
                    global_mass=global_mass,
                    reverse=False,
                )
                if self.use_mass
                else 0.0
            )
            # Energy loss (average of energy term from ODE step)
            loss_energy = e1.mean()

            # Combine losses with respective weights
            loss = (lambda_ot * loss_ot) + (lambda_mass * loss_mass) + (lambda_energy * loss_energy)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite combined loss during {stage_params['name']} "
                    f"forward interval {t0}->{t1}."
                )

            density_loss_value = 0.0
            pinn_loss_value = 0.0
            if use_density_loss:
                density_loss = density_fn(x1, data_t1, top_k=top_k)
                density_loss = density_loss.to(loss.device)
                loss += lambda_density * density_loss
                density_loss_value = self._history_float(density_loss)
                # print('density loss')
                # print(density_loss)
            if use_pinn_loss: 
                if 'lambda_pinn'  not in stage_params:
                    raise ValueError(
                        "When use_pinn_loss=True, 'lambda_pinn' must be provided in stage_params.(Default recommended (100))" 
                    )            
                lambda_pinn = stage_params['lambda_pinn'] 

                loss_pinn = calc_pinn_loss(self, t1, data_t1,sigma=stage_params['sigma'], use_mass=self.use_mass,trace_df_dz=trace_df_dz,device=self.device)
                # print("loss_pinn",loss_pinn)
                # print("loss",loss)
                loss += lambda_pinn * loss_pinn
                pinn_loss_value = self._history_float(loss_pinn)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Non-finite final loss during {stage_params['name']} "
                    f"forward interval {t0}->{t1}."
                )
            # print(f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy}), Density Loss: {density_loss:.4f} (λ={lambda_density})" if use_density_loss else f"OT Loss: {loss_ot:.4f} (λ={lambda_ot}), Mass Loss: {loss_mass:.4f} (λ={lambda_mass}), Energy Loss: {loss_energy:.4f} (λ={lambda_energy})", end="")
            # if use_pinn_loss:
            #     print(f", PINN Loss: {loss_pinn:.4f} (λ={lambda_pinn})")
            # Backpropagate gradients and update optimizer
            loss.backward()
            if max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    optimizer_parameters,
                    max_norm=float(max_grad_norm),
                    error_if_nonfinite=True,
                )
            self._optimizer_step()

            # Update initial state for next time interval (detach to avoid gradient accumulation)
            x0 = x1.clone().detach()
            lnw0 = lnw1.clone().detach()

            # Accumulate total loss over all time intervals
            total_loss += loss.item()
            component_sums['ot_loss'] += self._history_float(loss_ot)
            component_sums['mass_loss'] += self._history_float(loss_mass)
            component_sums['energy_loss'] += self._history_float(loss_energy)
            component_sums['density_loss'] += density_loss_value
            component_sums['pinn_loss'] += pinn_loss_value
            valid_intervals += 1

        # The released DeepRUOT scripts selected intermediate stage checkpoints
        # using the final forward interval's OT loss and captured the weights
        # before the reverse pass.  Keep that behavior opt-in so generic configs
        # retain their average-loss checkpointing semantics.
        forward_last_ot = float(loss_ot.detach().item())
        state_after_forward = None
        if stage_params.get('checkpoint_metric') == 'legacy_forward_last_ot':
            state_after_forward = copy.deepcopy(self.model.state_dict())
        if self.scheduler is not None and stage_params.get('scheduler_step_before_reverse', False):
            scheduler_metric = stage_params.get('scheduler_metric', 'forward_last_ot')
            scheduler_value = forward_last_ot if scheduler_metric == 'forward_last_ot' else total_loss / valid_intervals
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(scheduler_value)
            else:
                self.scheduler.step()

        # Optional reverse-time training to mirror legacy DeepRUOT behavior
        if self.config.get('reverse', False):
            rev_time_points = list(reversed(time_points))
            rev_data = list(reversed(data))
            relative_masses = [d.shape[0] / forward_mass0 for d in data]
            relative_masses_rev = list(reversed(relative_masses))
            reverse_mass_norm = stage_params.get('reverse_mass_norm', True)
            if reverse_mass_norm:
                denom = relative_masses_rev[0] if relative_masses_rev[0] != 0 else 1.0
                relative_masses_rev = [m / denom for m in relative_masses_rev]
            x0 = sample(rev_data[0], self.batch_size).to(self.device)
            lnw0 = torch.log(torch.ones(self.batch_size, 1) / self.batch_size).to(self.device)
            if stage_params.get('reverse_mass_offset', False):
                base_mass = relative_masses_rev[0] if relative_masses_rev else 1.0
                if isinstance(base_mass, torch.Tensor):
                    base_mass = base_mass.item()
                if base_mass > 0:
                    lnw0 = lnw0 + math.log(base_mass)

            for idx in range(1, len(rev_time_points)):
                self.optimizer.zero_grad()
                t0, t1 = rev_time_points[idx - 1], rev_time_points[idx]
                data_t1 = sample(rev_data[idx], self.batch_size).to(self.device)
                relative_mass = relative_masses_rev[idx]

                x1, lnw1, e1 = neural_ode_step(ode_func, x0, lnw0, t0, t1, self.device)

                if not torch.isfinite(x1).all() or not torch.isfinite(lnw1).all():
                    raise FloatingPointError(
                        f"Non-finite ODE state during {stage_params['name']} "
                        f"reverse interval {t0}->{t1}."
                    )
                try:
                    loss_ot = calc_ot_loss(
                        x1,
                        data_t1,
                        lnw1,
                        OT_loss_type,
                        alpha_spatial=alpha_spatial,
                        alpha_express=alpha_express,
                        spatial_dim=self.config.get('spatial_dim', 2),
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"OT loss failed during {stage_params['name']} "
                        f"reverse interval {t0}->{t1}."
                    ) from exc
                if not torch.isfinite(loss_ot):
                    raise FloatingPointError(
                        f"Non-finite OT loss during {stage_params['name']} "
                        f"reverse interval {t0}->{t1}."
                    )
                loss_mass = (
                    calc_mass_loss(
                        x1,
                        data_t1,
                        lnw1,
                        relative_mass,
                        global_mass=global_mass,
                        reverse=True,
                    )
                    if self.use_mass
                    else 0.0
                )
                loss_energy = e1.mean()
                loss = (lambda_ot * loss_ot) + (lambda_mass * loss_mass) - (lambda_energy * loss_energy)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite combined loss during {stage_params['name']} "
                        f"reverse interval {t0}->{t1}."
                    )

                density_loss_value = 0.0
                pinn_loss_value = 0.0
                if use_density_loss:
                    density_loss = density_fn(x1, data_t1, top_k=top_k)
                    density_loss = density_loss.to(loss.device)
                    loss += lambda_density * density_loss
                    density_loss_value = self._history_float(density_loss)
                if use_pinn_loss:
                    if 'lambda_pinn'  not in stage_params:
                        raise ValueError(
                            "When use_pinn_loss=True, 'lambda_pinn' must be provided in stage_params.(Default recommended (100))" 
                        )
                    lambda_pinn = stage_params['lambda_pinn']
                    loss_pinn = calc_pinn_loss(
                        self,
                        t1,
                        data_t1,
                        sigma=stage_params['sigma'],
                        use_mass=self.use_mass,
                        trace_df_dz=trace_df_dz,
                        device=self.device,
                    )
                    loss += lambda_pinn * loss_pinn
                    pinn_loss_value = self._history_float(loss_pinn)
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"Non-finite final loss during {stage_params['name']} "
                        f"reverse interval {t0}->{t1}."
                    )

                loss.backward()
                if max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        optimizer_parameters,
                        max_norm=float(max_grad_norm),
                        error_if_nonfinite=True,
                    )
                self._optimizer_step()

                x0 = x1.clone().detach()
                lnw0 = lnw1.clone().detach()
                total_loss += loss.item()
                component_sums['ot_loss'] += self._history_float(loss_ot)
                component_sums['mass_loss'] += self._history_float(loss_mass)
                component_sums['energy_loss'] += self._history_float(loss_energy)
                component_sums['density_loss'] += density_loss_value
                component_sums['pinn_loss'] += pinn_loss_value
                valid_intervals += 1

        # Return average loss per time interval (account for reverse pass if used)
        if valid_intervals != expected_intervals:
            raise RuntimeError(
                f"{stage_params['name']} completed only {valid_intervals}/"
                f"{expected_intervals} expected intervals."
            )
        average_loss = total_loss / expected_intervals
        self._last_neural_ode_epoch = {
            'forward_last_ot': forward_last_ot,
            'state_after_forward': state_after_forward,
            'mean_ot_loss': component_sums['ot_loss'] / expected_intervals,
            'mean_mass_loss': component_sums['mass_loss'] / expected_intervals,
            'mean_energy_loss': component_sums['energy_loss'] / expected_intervals,
            'mean_density_loss': (
                component_sums['density_loss'] / expected_intervals
            ),
            'mean_pinn_loss': component_sums['pinn_loss'] / expected_intervals,
            'n_intervals': int(valid_intervals),
        }
        return average_loss


    def run_flow_matching_stage(self, stage_params, data, time_points):
        """Execute training stage using Flow Matching mode
        
        Args:
            stage_params: Dictionary of parameters for current stage (epochs, sigma, etc.)
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        """
        # Create checkpoint directory for current stage
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)

        # Convert time points to tensor (device-compatible)
        time = torch.tensor(time_points, device=self.device, dtype=torch.float32)
        # Get sigma parameter for Flow Matching
        sigma = stage_params['sigma']
        # Get alpha regularization parameter (default to 1.0 if not specified)
        alpha_regm = stage_params.get('alpha_regm', 1.0)
        print("alpha_regm :", alpha_regm)
        self.sigma = sigma
        # Convert data to list of numpy arrays (required for compute_uot_plans)
        X = [data[i].float().cpu().detach().numpy() for i in range(len(time_points))]
        
        # Get flags for training different network components
        train_strategy = str(stage_params.get('train_strategy', 's')).lower()
        regress_v, regress_g, regress_score = 'v' in train_strategy, 'g' in train_strategy, 's' in train_strategy
        
        if regress_g or regress_v :
            uot_plans, sampling_info = compute_uot_plans(X, time_points,use_mini_batch_uot=True, chunk_size=1000, alpha_regm= alpha_regm ,reg_strategy="max_over_time")
        else :
            uot_plans, sampling_info = compute_uot_plans(X, time_points,use_mini_batch_uot=True, chunk_size=2000,reg_strategy='per_time')

        # Initialize Conditional Regularized Unbalanced Flow Matcher
        FM = ConditionalRegularizedUnbalancedFlowMatcher(sigma=sigma)
        # Get model saving strategy (default to 'best' if not specified)
        save_strategy = stage_params.get('save_strategy', 'best')
        # Initialize variables for tracking best model
        best_loss = float('inf')
        best_state_dict = None
        best_epoch = None
        last_recorded_epoch = None
        
        # Get batch size from stage parameters
        batch_size = stage_params['batch_size']



        # Training loop over epochs (with tqdm progress bar)
        for epoch in tqdm(
            range(1, stage_params['epochs'] + 1), desc='Flow matching'
        ):
            epoch_started_at = self._wall_time_start()
            optimizer_steps_before_epoch = getattr(
                self, '_optimizer_step_count', None
            )
            # Calculate loss for one epoch of Flow Matching training
            loss, penalty = self.train_flow_matching_epoch(
                FM, X, time,
                self.optimizer,
                stage_params['flow_matching']['lambda_penalty'],
                batch_size,
                uot_plans,
                sampling_info,
                regress_v, regress_g, regress_score,
            )

            # Stop training if loss becomes NaN (numerical instability)
            if torch.isnan(loss):
                self.logger.info("Training stopped due to NaN loss")
                # Load best model state before NaN occurred
                self.model.load_state_dict(best_state_dict)
                break

            # Update best model if current loss is lower than previous best
            loss_value = self._history_float(loss)
            penalty_value = self._history_float(penalty)
            total_loss_value = loss_value + penalty_value
            is_best = loss_value < best_loss
            if is_best:
                best_loss = loss_value
                best_state_dict = copy.deepcopy(self.model.state_dict())
                best_epoch = int(epoch)

            # Combine loss and penalty for backpropagation
            total_loss = loss + penalty
            # print("score_loss",loss)
            # print("penalty",penalty)

            total_loss.backward()
            # Update optimizer
            self._optimizer_step()
            last_recorded_epoch = int(epoch)
            self._record_training_epoch(
                stage_params=stage_params,
                epoch=epoch,
                loss=total_loss_value,
                checkpoint_metric='objective_loss',
                checkpoint_value=loss_value,
                is_best=is_best,
                epoch_wall_time_seconds=self._wall_time_elapsed(epoch_started_at),
                optimizer_steps_epoch=(
                    int(self._optimizer_step_count - optimizer_steps_before_epoch)
                    if optimizer_steps_before_epoch is not None
                    else None
                ),
                objective_loss=loss_value,
                penalty=penalty_value,
            )
            # Update scheduler if initialized
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(loss)
                else:
                    self.scheduler.step()

        # Determine which model state to save (best or last)
        if save_strategy == 'best':
            save_state = best_state_dict
            save_loss = best_loss
            selected_epoch = best_epoch
        else:  # 'last' strategy
            save_state = copy.deepcopy(self.model.state_dict())
            save_loss = total_loss_value
            selected_epoch = last_recorded_epoch

        # Load saved state (best or last) back to model
        self.model.load_state_dict(save_state)
        # Define checkpoint filename based on save strategy
        ckpt_filename = 'best_model.pth' if save_strategy == 'best' else 'last_model.pth'
        torch.save(save_state, os.path.join(ckpt_dir, ckpt_filename))
        self._mark_selected_checkpoint(selected_epoch)
        print(f"  {save_strategy.capitalize()} model (loss={save_loss:.4f}) "
              f"saved → {ckpt_dir}/{save_strategy}_model.pth")

    def train_flow_matching_epoch(self, FM, X, time,
                                  optimizer, lambda_pen, batch_size, uot_plans, sampling_info, regress_v, regress_g, regress_score):
        """Calculate loss for one epoch of Flow Matching training
        
        Args:
            FM: ConditionalRegularizedUnbalancedFlowMatcher instance
            X: List of numpy arrays where each element represents samples at a specific time point
            time: Tensor of time points (device-compatible)
            optimizer: Torch optimizer instance
            lambda_pen: Penalty weight for score network training
            batch_size: Batch size for sampling
            uot_plans: Precomputed UOT plans for sampling
            sampling_info: Additional sampling information from compute_uot_plans
            regress_v: Flag to train velocity network (v)
            regress_g: Flag to train growth network (g)
            regress_score: Flag to train score network
        
        Returns:
            tuple: (total_loss, penalty) where both are torch tensors
        """
        # Reset gradients before each batch
        optimizer.zero_grad()
        # Sample batch data for Flow Matching (time, positions, velocities, growth values, weights, noise)
        t, xt, ut, gt_samp, weights, eps = get_batch_uot_fm(FM, X, time, batch_size, uot_plans, sampling_info)
        # Reshape time tensor to (batch_size, 1) for concatenation with position data
        t = torch.unsqueeze(t, 1).to(self.device)

        # Compute lambda(t) (time-dependent weighting factor for score network)
        t_floor = torch.zeros_like(t)
        t_ceil = torch.zeros_like(t)
        # Determine time interval bounds (t_floor and t_ceil) for each sample in the batch
        for j in range(len(time) - 1):
            mask = (t >= time[j]) & (t < time[j + 1])
            t_floor[mask] = time[j]
            t_ceil[mask] = time[j + 1]
        # Calculate normalized time within interval and compute lambda(t)
        lambda_t = FM.compute_lambda((t - t_floor) / (t_ceil - t_floor))

        # Enable gradient computation for position data (required for score calculation via autograd)
        xt = xt.requires_grad_(True)

        # Initialize loss and penalty
        loss = 0.0
        penalty = 0.0
        # Train score network if enabled
        if regress_score:
            # Predict score potential (value_st) from score network
            value_st, st = self.model.compute_score(t=t, x=xt, create_graph=True)
            # Calculate weighted MSE loss for score network
            score_loss = torch.mean(weights * ((lambda_t[:, None] * st + eps) ** 2))
            # Handle NaN loss (set to 0 to avoid training instability)
            if torch.isnan(score_loss):
                score_loss = 0.0
            loss += score_loss
            # Add penalty term to regularize score potential (prevents exploding values)
            penalty += lambda_pen * torch.max(torch.relu(value_st))
        
        # Train velocity network (v) if enabled
        if regress_v:
            # Predict velocity from velocity network
            v_predict = self.model.predict_velocity(t=t, x=xt)
            # Add weighted MSE loss between predicted and target velocities
            loss += torch.mean(weights * (v_predict - ut) ** 2)
        
        # Train growth network (g) if enabled
        if regress_g:
            # Predict growth values from growth network
            g_predict = self.model.predict_growth(t=t, x=xt)
            # Add weighted MSE loss between predicted and target growth values (scaled by 1000 for better convergence)
            loss += 1000 * torch.mean(weights * (g_predict - gt_samp) ** 2)

        return torch.as_tensor(loss, device=self.device), torch.as_tensor(penalty, device=self.device)

    def _generate_score_trajectory(self, data, time_points):
        """Generate a trajectory with interaction for SF2M-style score training."""
        X = [d.detach().cpu().numpy() for d in data]
        n_times = len(time_points)
        base_size = X[0].shape[0]
        batch_size = min(base_size, 1024)
        if base_size > batch_size:
            indices = np.random.choice(base_size, size=batch_size, replace=False)
            x0 = torch.from_numpy(X[0][indices]).float().to(self.device)
        else:
            x0 = torch.from_numpy(X[0]).float().to(self.device)
        lnw0 = torch.log(torch.ones(batch_size, 1, device=self.device) / batch_size)
        trajectory = [x0]
        ode_func = ODEFunc2InteractionEnergy(self.model, use_mass=self.use_mass)

        for t_start in range(n_times - 1):
            t_mid = torch.tensor([time_points[t_start], time_points[t_start + 1]], device=self.device, dtype=torch.float32)
            lnw0.requires_grad = True
            m0 = torch.zeros_like(lnw0)
            initial_state_energy = (trajectory[-1], lnw0, m0)
            xtt, _, _ = odeint(
                ode_func,
                initial_state_energy,
                t_mid,
                method='euler',
                options=dict(step_size=0.1),
            )
            trajectory.append(xtt[-1].detach())
        return trajectory

    def run_score_matching_stage(self, stage_params, data, time_points):
        """Train score network using SF2M-style score matching."""
        ckpt_dir = os.path.join(self.config.get('ckpt_dir', '.'), stage_params['name'])
        os.makedirs(ckpt_dir, exist_ok=True)

        time = torch.tensor(time_points, device=self.device, dtype=torch.float32)
        X = [data[i].float().cpu().detach().numpy() for i in range(len(time_points))]
        batch_size = stage_params['batch_size']
        sigma = stage_params['sigma']
        lambda_penalty = stage_params.get('lambda_penalty', 0.0)

        FM = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        trajectory = self._generate_score_trajectory(data, time_points)

        best_loss = float('inf')
        best_state_dict = None
        best_epoch = None
        last_loss = None

        for epoch in tqdm(
            range(1, stage_params['epochs'] + 1), desc='Score matching'
        ):
            epoch_started_at = self._wall_time_start()
            optimizer_steps_before_epoch = getattr(
                self, '_optimizer_step_count', None
            )
            self.optimizer.zero_grad()
            t, xt, _, eps = get_batch_size(FM, X, trajectory, batch_size, time, return_noise=True)
            t = torch.unsqueeze(t, 1).to(self.device)

            t_floor = torch.zeros_like(t)
            t_ceil = torch.zeros_like(t)
            for j in range(len(time) - 1):
                mask = (t >= time[j]) & (t < time[j + 1])
                t_floor[mask] = time[j]
                t_ceil[mask] = time[j + 1]
            lambda_t = FM.compute_lambda((t - t_floor) / (t_ceil - t_floor))

            xt = xt.requires_grad_(True)
            value_st, st = self.model.compute_score(t=t, x=xt, create_graph=True)
            score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
            if not torch.isfinite(score_loss):
                raise FloatingPointError(
                    f"Non-finite score loss during {stage_params['name']} "
                    f"at epoch {epoch}."
                )

            penalty = lambda_penalty * torch.max(torch.relu(value_st))
            loss = score_loss + penalty
            score_loss_value = self._history_float(score_loss)
            penalty_value = self._history_float(penalty)
            loss_value = self._history_float(loss)
            is_best = loss_value < best_loss
            if is_best:
                best_loss = loss_value
                best_state_dict = copy.deepcopy(self.model.score_net.state_dict())
                best_epoch = int(epoch)

            loss.backward()
            self._optimizer_step()
            last_loss = loss_value
            self._record_training_epoch(
                stage_params=stage_params,
                epoch=epoch,
                loss=loss_value,
                checkpoint_metric='total_loss',
                checkpoint_value=loss_value,
                is_best=is_best,
                epoch_wall_time_seconds=self._wall_time_elapsed(epoch_started_at),
                optimizer_steps_epoch=(
                    int(self._optimizer_step_count - optimizer_steps_before_epoch)
                    if optimizer_steps_before_epoch is not None
                    else None
                ),
                score_loss=score_loss_value,
                penalty=penalty_value,
            )

        save_strategy = stage_params.get('save_strategy', 'best')
        if save_strategy == 'best':
            save_state = best_state_dict
            save_loss = best_loss
            selected_epoch = best_epoch
        elif save_strategy == 'last':
            save_state = copy.deepcopy(self.model.score_net.state_dict())
            save_loss = last_loss
            selected_epoch = int(epoch) if last_loss is not None else None
        else:
            raise ValueError("score-matching save_strategy must be 'best' or 'last'.")
        if save_state is None or save_loss is None:
            raise RuntimeError(f"{stage_params['name']} produced no finite score checkpoint.")
        self.model.score_net.load_state_dict(save_state)
        torch.save(save_state, os.path.join(ckpt_dir, 'score_model.pth'))
        self._mark_selected_checkpoint(selected_epoch)
        print(
            f"  {save_strategy.capitalize()} score model (loss={save_loss:.4f}) "
            f"saved → {ckpt_dir}/score_model.pth"
        )

    def evaluate(self,adata, data, time_points):
        """Evaluate trained model using Wasserstein-1 distance and Total Mass Variation (TMV)
        
        Args:
            data: List of tensors where each element represents samples at a specific time point
            time_points: List of time values corresponding to each element in 'data'
        
        Returns:
            list: List of Wasserstein-1 distances for each time point (excluding initial time)
        """
        print(f"\n--- Starting Evaluation ---")
        device = self.device
        # Align evaluation order to increasing time points
        if self.config.get('reverse', False):
            data_by_time = {t: data[i] for i, t in enumerate(time_points)}
            eval_time_points = sorted(time_points)
            eval_data = [data_by_time[t] for t in eval_time_points]
        else:
            eval_time_points = time_points
            eval_data = data

        # Get initial time point data (t=0)
        x0 = eval_data[0].to(device)
        # Freeze model parameters during evaluation (disable gradient computation)
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Get sigma parameter (use stored value or default to 0.05 if not available)
        sigma = getattr(self, 'sigma', None) or 0.05

        # Simulate trajectory using the trained model
        point, weight = simulate_trajectory(
            adata,
            self.model,
            x0,
            sigma,           
            eval_time_points,
            dt=0.01,  # Time step for ODE simulation
            device=x0.device
        )

        # Calculate Wasserstein-1 distance for each time point (excluding initial time)
        wasserstein_scores = []
        for idx in range(1, len(eval_time_points)):
            t0, t1 = eval_time_points[0], eval_time_points[idx]
            # Get target data at current time point (convert to numpy for OT computation)
            data_t1 = eval_data[idx].detach().cpu().numpy()
            # Get predicted positions and weights from simulated trajectory
            x1 = point[idx]
            m1 = weight[idx]

            # Calculate Total Mass Variation (TMV) between predicted and true mass
            tmv = np.abs(m1.sum() - eval_data[idx].shape[0] / eval_data[0].shape[0])
            # Normalize predicted weights to sum to 1 (required for OT)
            m1 = m1 / m1.sum()

            # Create uniform weights for target data (sum to 1)
            m2 = np.ones(data_t1.shape[0]) / data_t1.shape[0]
            # Compute Euclidean distance matrix between target and predicted points
            cost_matrix = ot.dist(data_t1, x1, metric='euclidean')

            # Calculate Wasserstein-1 distance using Earth Mover's Distance (EMD)
            w1 = ot.emd2(
                m2,
                m1.reshape(-1),  # Reshape to 1D array (required by ot.emd2)
                cost_matrix,
                numItermax=1e7  # Increase max iterations for convergence
            )

            # Store results and print progress
            wasserstein_scores.append(w1)
            print(f"  Time Point {t1}: Wasserstein-1 Distance = {w1:.4f}")
            print(f"  Time Point {t1}: TMV = {tmv:.4f}")
        
        return wasserstein_scores

    
    # def generate_state_trajectory(self, data, time_points):
    #     """Generate reference trajectory without score guidance (using only velocity and growth components)
        
    #     Args:
    #         data: List of tensors where each element represents samples at a specific time point
    #         time_points: List of time values corresponding to each element in 'data'
        
    #     Returns:
    #         list: List of tensors representing predicted positions at each time point (detached from graph)
    #     """
    #     # Get initial time point data (t=0)
    #     x0 = data[0].to(self.device)
    #     n_samples = x0.shape[0]

    #     # Initialize ODE function (force disable score component)
    #     ode_func = ODEFunc(
    #         model=self.model,
    #         sigma=self.config['training']['defaults'].get('sigma', 0.1),
    #         use_mass=self.use_mass,
    #         score_use=False,
    #         score_flow_matching_use=False
    #     ).to(self.device)

    #     # ODE initial state: (positions, log-weights, mass)
    #     init_lnw = torch.log(torch.ones(n_samples, 1) / n_samples).to(self.device)
    #     init_m = torch.zeros_like(init_lnw).to(self.device)
    #     initial_state = (x0, init_lnw, init_m)

    #     # Solve ODE to get full trajectory
    #     t_eval = torch.tensor(time_points, device=self.device, dtype=torch.float32)
    #     traj_x, _, _ = odeint(
    #         func=ode_func,
    #         y0=initial_state,
    #         t=t_eval,
    #         method='euler'  # Euler method for ODE solving (fast but less accurate)
    #     )

    #     # Split trajectory by time point and detach from computation graph (avoid memory leaks)
    #     return [traj_x[i].detach() for i in range(len(time_points))]


    # def generate_state_trajectory1(self, data, time_points, reg=None, reg_m=None, method='sinkhorn', numItermax=1000,
    #                                stopThr=1e-6, **kwargs):
    #     """Generate trajectory with Unbalanced Sinkhorn matching (fixed: ensure valid trajectory output + add error handling)
        
    #     Args:
    #         data: List of tensors where each element represents samples at a specific time point
    #         time_points: List of time values corresponding to each element in 'data'
    #         reg: Regularization parameter for Sinkhorn (auto-calculated if None)
    #         reg_m: Mass regularization parameter for Unbalanced Sinkhorn (auto-calculated if None)
    #         method: Matching method (default: 'sinkhorn')
    #         numItermax: Maximum number of iterations for Sinkhorn
    #         stopThr: Convergence threshold for Sinkhorn
    #         **kwargs: Additional keyword arguments
        
    #     Returns:
    #         list: List of tensors representing matched trajectory (detached from graph)
    #     """
    #     try:
    #         # 1. Get number of samples at each time point
    #         max_iter = numItermax
    #         tol = stopThr
    #         data_sizes = [d.shape[0] for d in data]
    #         raw_masses = data_sizes

    #         # 2. Determine sample sizes for each time point (balance between speed and accuracy)
    #         min_size = min(data_sizes)
    #         max_size = max(data_sizes)
    #         print("min_size", min_size, "max_size", max_size)
            
    #         if min_size >= 1024:
    #             # Scale sample sizes proportionally if minimum size is ≥1024
    #             sample_sizes = [max(1, int(round(1024 * s / min_size))) for s in data_sizes]
    #         elif max_size >= 1024:
    #             # Cap sample sizes at max_size if maximum size is ≥1024 (avoid oversampling)
    #             sample_sizes = []
    #             for s in data_sizes:
    #                 target = max(1, int(round(1024 * s / min_size)))
    #                 target = min(target, s)
    #                 sample_sizes.append(target)
    #         else:
    #             # Use original sample sizes if all are <1024
    #             sample_sizes = data_sizes

    #         # 3. Sample data for each time point (ensure consistent batch size)
    #         sampled_data = []
    #         for t_idx in range(len(time_points)):
    #             size = sample_sizes[t_idx]
    #             data_t = data[t_idx].to(self.device)
    #             # Oversample if current time point has fewer samples than target size
    #             if data_t.shape[0] < size:
    #                 indices = torch.randint(0, data_t.shape[0], (size,), device=self.device)
    #             else:
    #                 # Undersample if current time point has more samples than target size
    #                 indices = torch.randperm(data_t.shape[0], device=self.device)[:size]
    #             sampled = data_t[indices]
    #             sampled_data.append(sampled)

    #         # 4. Unbalanced Sinkhorn matching to connect time points
    #         matched_trajectories = [sampled_data[0]]  # Initialize trajectory with first time point

    #         # Iterate over time points to match consecutive time steps
    #         for t_idx in range(1, len(time_points)):
    #             t_prev = time_points[t_idx - 1]
    #             t_curr = time_points[t_idx]
    #             prev_points = matched_trajectories[-1]  # Points from previous time point
    #             curr_points = sampled_data[t_idx]  # Points from current time point
    #             n, m = prev_points.shape[0], curr_points.shape[0]

    #             # Convert tensors to numpy arrays (required for OT library)
    #             prev_np = prev_points.cpu().detach().numpy()
    #             curr_np = curr_points.cpu().detach().numpy()

    #             # Get true mass values for previous and current time points
    #             prev_mass = raw_masses[t_idx - 1]
    #             curr_mass = raw_masses[t_idx]

    #             # Auto-calculate regularization parameters if not provided
    #             if reg is None or reg_m is None:
    #                 auto_reg, auto_reg_m = self.calculate_auto_regularization(prev_np, curr_np, prev_mass, curr_mass)
    #                 reg = auto_reg if reg is None else reg
    #                 reg_m = auto_reg_m if reg_m is None else reg_m
    #                 print(f"Auto-calculated regularization: reg={reg:.4f}, reg_m={reg_m:.4f}")
    #             else:
    #                 print(f"User-specified regularization: reg={reg:.4f}, reg_m={reg_m:.4f}")

    #             # Predict source weights (a) using growth network (fixed: ensure correct weight calculation)
    #             # a. Initialize log-weights for previous time point (uniform distribution)
    #             lnw_prev_init = torch.log(torch.ones(n, 1, device=self.device) / n)
    #             # b. Define time interval for ODE solving (from previous to current time point)
    #             t_interval = torch.tensor([t_prev, t_curr], device=self.device, dtype=torch.float32)
    #             # c. Initialize ODE state (positions, log-weights, mass)
    #             initial_state = (prev_points, lnw_prev_init, torch.zeros_like(lnw_prev_init, device=self.device))
    #             # d. Solve ODE to get predicted log-weights at current time point
    #             traj_x, traj_lnw, _ = odeint(
    #                 func=self.ode_func,
    #                 y0=initial_state,
    #                 t=t_interval,
    #                 method='euler'
    #             )
    #             # e. Convert log-weights to probabilities (normalize to sum to 1)
    #             lnw_prev_pred = traj_lnw[-1]
    #             mu_prev = torch.exp(lnw_prev_pred)
    #             mu_prev = mu_prev / mu_prev.sum()
    #             a = mu_prev.cpu().detach().numpy().squeeze()  # Source weights (1D array)

    #             # Target weights (b): uniform distribution over current time point samples
    #             nu_curr = torch.ones(m, 1, device=self.device) / m
    #             b = nu_curr.cpu().detach().numpy().squeeze()  # Target weights (1D array)

    #             # Compute Euclidean distance matrix between previous and current points
    #             M = ot.dist(prev_np, curr_np)
    #             # Solve Unbalanced Sinkhorn to get transport matrix
    #             transport_matrix = ot.unbalanced.sinkhorn_unbalanced(
    #                 a, b, M, reg, reg_m,
    #                 numItermax=max_iter, stopThr=tol
    #             )

    #             # Match current time point points to previous time point (max weight in transport matrix)
    #             sinkhorn_result = torch.tensor(transport_matrix, device=self.device)
    #             matched_indices = torch.argmax(sinkhorn_result, dim=1)  # For each previous point, find best current point
    #             matched_points = curr_points[matched_indices]
    #             matched_trajectories.append(matched_points)

    #         # Ensure trajectory is not empty (raise error if no points were generated)
    #         if not matched_trajectories:
    #             raise ValueError("Trajectory generation failed: no points were generated")

    #         # Detach all points from computation graph and return trajectory
    #         trajectory = [points.detach() for points in matched_trajectories]
    #         return trajectory

    #     except Exception as e:
    #         # Print error message and return original data (detached) as fallback
    #         print(f"Trajectory generation encountered an error: {str(e)}")
    #         return [data[t_idx].detach() for t_idx in range(len(time_points))]


    # def visualize_trajectory(self, trajectory, trajectory_times):
    #     """Visualize trajectory with scatter plots (time-colored) and connecting lines for each trajectory chain
        
    #     Args:
    #         trajectory: List of tensors where each element represents predicted positions at a specific time point
    #         trajectory_times: List of time values corresponding to each element in 'trajectory'
    #     """
    #     import matplotlib.pyplot as plt
    #     import numpy as np

    #     # 1. Prepare data for scatter plot (combine all time points)
    #     all_data = np.concatenate([x.cpu().detach().numpy() for x in trajectory], axis=0)
    #     # Create time labels for color coding (each sample gets its corresponding time point)
    #     time_labels = np.concatenate([np.full(x.shape[0], t.item()) for x, t in zip(trajectory, trajectory_times)])

    #     # 2. Create plot and scatter plot (time-colored points)
    #     plt.figure(figsize=(8, 6))
    #     # Scatter plot: color by time point, semi-transparent, higher z-order (on top of lines)
    #     scatter = plt.scatter(
    #         all_data[:, 0],
    #         all_data[:, 1],
    #         c=time_labels,
    #         cmap='viridis',
    #         alpha=0.6,
    #         zorder=2
    #     )
    #     # Add color bar to indicate time point mapping
    #     plt.colorbar(scatter, label='Time Point')

    #     # 3. Add connecting lines for each trajectory chain (same sample across time points)
    #     # Reshape trajectory to (num_time_points, num_trajectories, 2) for easy indexing
    #     traj_matrix = np.concatenate([
    #         pts.cpu().detach().numpy()[:, :2][None, ...]  # Shape: (1, num_trajectories, 2)
    #         for pts in trajectory
    #     ], axis=0)  # Final shape: (num_time_points, num_trajectories, 2)
    #     T, n_traj = traj_matrix.shape[:2]

    #     # Plot line for each trajectory chain (low alpha + low z-order to not obscure scatter points)
    #     for traj_id in range(n_traj):
    #         plt.plot(
    #             traj_matrix[:, traj_id, 0],  # X-coordinates across time
    #             traj_matrix[:, traj_id, 1],  # Y-coordinates across time
    #             color='black',
    #             linewidth=0.8,
    #             alpha=0.4,
    #             zorder=1
    #         )

    #     # Add plot labels and title
    #     plt.xlabel('Latent Dimension 1')
    #     plt.ylabel('Latent Dimension 2')
    #     plt.title('Trajectory Visualization (lines connect same chain)')
    #     # Save plot (high DPI for clarity, tight layout to avoid label cutoff)
    #     plt.savefig("/home/sjt/workspace2/CytoBridge_test_main/figures/tra_test.png", dpi=300, bbox_inches='tight')


    # def _plot_snapshot(self, epoch, stage_params, data, time_points, exp_fig_dir):
    #     """Plot SDE trajectory and score field at specific time points for current epoch
        
    #     Args:
    #         epoch: Current training epoch (for filename labeling)
    #         stage_params: Stage-specific parameters (not used directly but kept for consistency)
    #         data: Input data (not used directly but kept for consistency)
    #         time_points: List of time points (not used directly but kept for consistency)
    #         exp_fig_dir: Directory to save plot files
    #     """
    #     # Create directory for figures if it doesn't exist
    #     os.makedirs(exp_fig_dir, exist_ok=True)

    #     # 3. Plot score field for time points t=0,1,2,3,4
    #     for t in [0, 1, 2, 3, 4]:
    #         # Define save path with epoch and time point labels
    #         save_score = os.path.join(exp_fig_dir, f"score_epoch{epoch}_t{t}.png")
    #         # Generate and save score field plot
    #         plot_score_and_gradient(
    #             dynamical_model=self.model,
    #             device=self.device,
    #             t_value=float(t),  # Time point to visualize
    #             x_range=(0, 2.5),  # X-axis range for grid
    #             y_range=(0, 2.5),  # Y-axis range for grid
    #             save_path=save_score,
    #             cmap='rainbow'  # Color map for score visualization
    #         )
