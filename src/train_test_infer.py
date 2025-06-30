from lightning import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from logger.mlflow_logger import mlflow_logger_setup
from modeling.lit_networks import LitAAE
from src.dataset import AnomalyDataModule


def model_setup(args):
    """setup lightning model."""
    params = {
        "class_name": args.class_name,
        "learning_rate": args.learning_rate,
        "disc_learning_rate": args.disc_learning_rate,
        "recon_weights": args.recon_weights,
        "backbone": args.backbone,
        "latent_dim": args.latent_dim,
        "image_size": args.image_size,
    }
    return LitAAE(**params)


def data_module_setup(args):
    """Setup datamodule."""
    params = {
        "root_dir": args.root_dir,
        "batch_size": args.batch_size,
        "class_name": args.class_name,
        "image_size": args.image_size,
        "image_resize": args.image_resize,
        "rand_aug": args.rand_aug,
    }
    return AnomalyDataModule(**params)


def trainer_setup(args):
    """Create trainer configuration based on mode."""
    mlflow_logger = mlflow_logger_setup(args)

    base_config = {
        "logger": mlflow_logger,
        "accelerator": "auto",
        "precision": "16-mixed",
    }

    if args.mode == "train":
        monitor_value = "train_recon_loss"
        callbacks = [
            ModelCheckpoint(
                dirpath="checkpoints",
                filename=f"{args.backbone}_{args.class_name}",
                save_top_k=1,
                monitor=monitor_value,
            ),
            EarlyStopping(
                monitor=monitor_value, 
                min_delta=1e-3, 
                patience=40, 
                verbose=False, 
                mode="min"
            ),
            LearningRateMonitor(logging_interval="epoch")
        ]
        
        train_config = {
            "max_epochs": args.epochs,
            "callbacks": callbacks,
            "enable_progress_bar": True,
            "log_every_n_steps": 5,
        }
        base_config.update(train_config)
    
    return Trainer(**base_config)


def run(args):
    """
    Unified pipeline function that handles train/test/infer based on mode.
    
    Args:
        args: Configuration arguments
        mode: "train", "test", or "infer"
    """
    datamodule = data_module_setup(args)
    if args.mode == "train":
        datamodule.setup("fit")    
    lit_model = model_setup(args)
    trainer = trainer_setup(args)
    
    if args.mode == "train":
        if args.checkpoint:
            print(f"Resuming training from checkpoint: {args.checkpoint}")
        else:
            print(f"Training {args.backbone} model on {args.class_name} class.")
        trainer.fit(
            lit_model, 
            train_dataloaders=datamodule.train_dataloader(),
            ckpt_path=args.checkpoint
        )
                  
    elif args.mode == "test":
        print(f"Testing model from checkpoint: {args.checkpoint}")
        trainer.test(lit_model, datamodule=datamodule, ckpt_path=args.checkpoint)

    elif args.mode == "infer":
        print(f"Running inference from checkpoint: {args.checkpoint}")
        trainer.predict(lit_model, datamodule=datamodule, ckpt_path=args.checkpoint)
        
    else:
        raise ValueError(f"Unknown mode: {args.mode}. Use 'train', 'test', or 'infer'")
    