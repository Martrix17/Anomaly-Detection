from lightning.pytorch.loggers import MLFlowLogger


def mlflow_logger_setup(args):
    mlflow_logger = MLFlowLogger(
        experiment_name="Anomaly-Detection",
        tracking_uri="mlruns",
        run_name=f"{args.mode}-{args.backbone}-{args.class_name}",
    )
    mlflow_logger.log_hyperparams(
        {
            "learning_rate": args.learning_rate,
            "disc_learning_rate": args.disc_learning_rate,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "latent_dim": args.latent_dim,
            "image_size": args.image_size,
            "recon_weights": args.recon_weights,
        }
    )
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "dataset", args.root_dir)
    mlflow_logger.experiment.set_tag(mlflow_logger.run_id, "backbone", args.backbone)
    mlflow_logger.experiment.set_tag(
        mlflow_logger.run_id, "class_name", args.class_name
    )
    return mlflow_logger
