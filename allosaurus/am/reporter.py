from allosaurus.model import get_model_path

class Reporter:

    def __init__(self, train_config):
        self.train_config = train_config

        self.model_path = get_model_path(train_config.new_model)

        # whether write into std
        self.verbose = train_config.verbose

        # log file
        self.log_file = None

        # Weights & Biases run (disabled when project is "none")
        self.wandb_run = None

        self.open()

    def open(self):
        # whether write into log file
        self.log_file = None
        if self.train_config.log != 'none':
            self.log_file = open(self.model_path / 'log.txt', 'w', encoding='utf-8')

        wandb_project = getattr(self.train_config, 'wandb_project', 'none')
        if wandb_project != 'none':
            try:
                import wandb
            except ImportError as exc:
                raise RuntimeError("wandb logging requested, but wandb is not installed") from exc
            self.wandb_run = wandb.init(
                project=wandb_project,
                entity=getattr(self.train_config, 'wandb_entity', None) or None,
                name=getattr(self.train_config, 'wandb_run_name', None) or None,
                mode=getattr(self.train_config, 'wandb_mode', 'online'),
                config=vars(self.train_config),
            )

    def close(self):
        if self.log_file:
            self.log_file.close()
        if self.wandb_run:
            self.wandb_run.finish()

    def log_metrics(self, metrics):
        if self.wandb_run:
            self.wandb_run.log(metrics)

    def write(self, message):

        if self.verbose:
            print(message)

        if self.log_file:
            self.log_file.write(message+'\n')
            self.log_file.flush()
