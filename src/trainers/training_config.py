import pathlib


class TrainingConfig:
    def __init__(
        self,
        output_dir: str,
        num_train_epochs: int,
        learning_rate: float,
        betas: tuple[float, float],
        weight_decay: float,
        train_batch_size: int,
        label_smoothing: float = 0.0,
        save_strategy: str | None = None,
        save_steps: int | None = None,
        eval_batch_size: int | None = None,
        eval_strategy: str | None = None,
    ):
        self.output_dir = pathlib.Path(output_dir).resolve()
        self.num_train_epochs = num_train_epochs
        self.learning_rate = learning_rate
        self.betas = betas
        self.weight_decay = weight_decay
        self.train_batch_size = train_batch_size
        self.label_smoothing = label_smoothing
        self.save_strategy = save_strategy
        self.save_steps = save_steps
        self.eval_batch_size = eval_batch_size
        self.eval_strategy = eval_strategy
