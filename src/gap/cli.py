"""Command-line interface for the GAP genetic ancestry prediction pipeline."""

import argparse
import os
from typing import Optional

import pandas as pd

from .ancestry import Ancestry
from .models import GapLearn


def parse_float_or_none(value: Optional[str]) -> Optional[float]:
    """Parse a CLI argument as a float, treating ``"none"`` as ``None``.

    Args:
        value: Raw string value from the command line, or ``None``.

    Returns:
        The parsed float, or ``None`` if ``value`` is ``None`` or the string
        ``"none"`` (case-insensitive).

    Raises:
        argparse.ArgumentTypeError: If ``value`` cannot be parsed as a float.
    """
    if value is None or str(value).lower() == "none":
        return None
    try:
        return float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid float value: '{value}'")


def get_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser for the ``gap`` CLI.

    Returns:
        A configured :class:`argparse.ArgumentParser` with one subparser per
        pipeline step.
    """
    formatter_class = argparse.ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(formatter_class=formatter_class)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p1 = subparsers.add_parser(
        "get-reference-data", help="get reference data with ancestry labeled"
    )
    p1.add_argument(
        "--source", type=str, default="1000genomes", help="data source to download from"
    )
    p1.add_argument(
        "--output_dir",
        type=str,
        default="ref",
        help="output directory to save the downloaded data",
    )

    p2 = subparsers.add_parser(
        "merge-dataset-with-reference", help="merge your dataset with the reference data"
    )
    p2.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="your input vcf file, must be on GRCh38 build and the REF allele must be "
        "consistent with the reference genome",
    )
    p2.add_argument(
        "--reference",
        type=str,
        default="ref/1000genomes_unrelated.vcf.gz",
        help="the vcf file of the reference data downloaded previously",
    )
    p2.add_argument(
        "--output", type=str, default="data/merged", help="the merged file, prefix only"
    )
    p2.add_argument(
        "--r2_threshold",
        type=parse_float_or_none,
        default=None,
        help="the R2 threshold for filtering the input dataset based on imputation "
        "quality, skip filtering if set to None",
    )
    p2.add_argument("--threads", type=int, default=2, help="number of threads to use")

    p3 = subparsers.add_parser(
        "feature-engineering", help="feature engineering for ancestry prediction"
    )
    p3.add_argument(
        "--input",
        type=str,
        default="data/merged",
        help="the merged file from the previous step, prefix only",
    )
    p3.add_argument(
        "--output",
        type=str,
        default="data/features.txt",
        help="the output feature matrix file",
    )
    p3.add_argument(
        "--pruning",
        type=str,
        default="True",
        help="remove variants that are highly correlated",
    )
    p3.add_argument(
        "--pruning_params",
        type=str,
        default="50,5,0.2",
        help="window size, step size, r2 threshold for pruning",
    )
    p3.add_argument(
        "--pca", type=str, default="True", help="dimensionality reduction using PCA"
    )
    p3.add_argument(
        "--n_pcs", type=int, default=50, help="first n principal components to keep"
    )
    p3.add_argument("--threads", type=int, default=2, help="number of threads to use")

    p4 = subparsers.add_parser(
        "add-labels", help="add ancestry labels to the feature matrix"
    )
    p4.add_argument(
        "--feature_file",
        type=str,
        default="data/features.txt",
        help="the feature matrix file",
    )
    p4.add_argument(
        "--label_file",
        type=str,
        default="ref/1000genomes_unrelated_sampleInfo.txt",
        help="the file containing ancestry labels for the reference samples",
    )

    p5 = subparsers.add_parser(
        "split-train-test", help="split the dataset into train and test sets"
    )
    p5.add_argument(
        "--input",
        type=str,
        default="data/features.txt",
        help="the input file with features and labels",
    )
    p5.add_argument(
        "--test_size",
        type=float,
        default=0.2,
        help="proportion of the dataset to include in the test split",
    )

    p6 = subparsers.add_parser(
        "train-model", help="train models with different parameters"
    )
    p6.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="the configuration file specifying models and parameters",
    )
    p6.add_argument(
        "--input",
        type=str,
        default="data/features.txt",
        help="the input feature matrix file",
    )
    p6.add_argument(
        "--task",
        type=str,
        default="Superpopulation",
        help="the prediction task, e.g., Superpopulation or Population",
    )
    p6.add_argument(
        "--conditional",
        type=str,
        default="True",
        help="if training conditional models when predict Population",
    )
    p6.add_argument(
        "--metrics_file",
        type=str,
        default="data/metrics.txt",
        help="the output file to save model performance metrics",
    )
    p6.add_argument(
        "--n_features",
        type=str,
        default="10,20,30",
        help="the number of features to use for training",
    )

    p7 = subparsers.add_parser(
        "eval-model", help="evaluate models and select the best one"
    )
    p7.add_argument(
        "--config_file",
        type=str,
        default="config.yaml",
        help="the configuration file specifying models and parameters",
    )
    p7.add_argument(
        "--input",
        type=str,
        default="data/features.txt",
        help="the input feature matrix file",
    )
    p7.add_argument(
        "--task",
        type=str,
        default="Superpopulation",
        help="the prediction task, e.g., Superpopulation or Population",
    )
    p7.add_argument(
        "--conditional",
        type=str,
        default="True",
        help="if training conditional models when predict Population",
    )
    p7.add_argument(
        "--metrics_file",
        type=str,
        default="data/metrics.txt",
        help="the model performance metrics from the training step",
    )
    p7.add_argument(
        "--model_file",
        type=str,
        default="data/model.pkl",
        help="the output file containing the trained model",
    )

    p8 = subparsers.add_parser(
        "predict", help="predict ancestry using the trained model"
    )
    p8.add_argument(
        "--input",
        type=str,
        default="data/ToPredict.txt",
        help="the input feature matrix file for prediction",
    )
    p8.add_argument(
        "--output",
        type=str,
        default="data/Predicted.txt",
        help="the output file to save the predictions",
    )
    p8.add_argument(
        "--task",
        type=str,
        default="Superpopulation",
        help="the prediction task, e.g., Superpopulation or Population",
    )
    p8.add_argument(
        "--conditional",
        type=str,
        default="True",
        help="if using the conditional models for Population",
    )
    p8.add_argument(
        "--metrics_file",
        type=str,
        default="data/metrics.txt",
        help="the model performance metrics from the training step",
    )
    p8.add_argument(
        "--model_file", type=str, default="data/model.pkl", help="the saved model file"
    )
    p8.add_argument(
        "--label_file",
        type=str,
        default="data/labels.txt",
        help="the model performance metrics from the training step",
    )
    p8.add_argument(
        "--model_name",
        type=str,
        default=None,
        help="the model name to use for prediction, using the best model if not specified",
    )

    p9 = subparsers.add_parser(
        "summarize",
        help="get a summary table with both Superpopulation and Population predictions",
    )
    p9.add_argument(
        "--input_sp",
        type=str,
        default="data/Predicted_Superpopulation.txt",
        help="the predicted results for Superpopulation",
    )
    p9.add_argument(
        "--input_p",
        type=str,
        default="data/Predicted_Population.txt",
        help="the predicted results for Population",
    )
    p9.add_argument(
        "--conditional",
        type=str,
        default="True",
        help="if using the conditional models for Population",
    )
    p9.add_argument(
        "--outfile",
        type=str,
        default="GenetcicAncestry.txt",
        help="the output summary file combining both predictions",
    )

    p10 = subparsers.add_parser(
        "format-table", help="convert the table to a specific format, internal use only"
    )
    p10.add_argument(
        "--input",
        type=str,
        default="GenetcicAncestry.txt",
        help="the input summary file",
    )
    p10.add_argument(
        "--labels_dir",
        type=str,
        default="data",
        help="the directory containing the label files",
    )
    p10.add_argument("--fmt", type=str, default="IIDP", help="the output format")

    return parser


def main() -> None:
    """Parse command-line arguments and dispatch to the requested GAP subcommand."""
    parser = get_parser()
    args = parser.parse_args()
    an = Ancestry()
    if args.command == "get-reference-data":
        an.get_reference_data(source=args.source, out_dir=args.output_dir)
    elif args.command == "merge-dataset-with-reference":
        an.merge_dataset_with_reference(
            dataset_vcf=args.dataset,
            reference_vcf=args.reference,
            out_file=args.output,
            r2_threshold=args.r2_threshold,
            threads=args.threads,
        )
    elif args.command == "feature-engineering":
        pruning = args.pruning.lower() == "true"
        pruning_params = [float(x) for x in args.pruning_params.split(",")]
        pca = args.pca.lower() == "true"
        an.feature_engineering(
            in_file=args.input,
            out_file=args.output,
            pruning=pruning,
            pruning_params=pruning_params,
            pca=pca,
            n_pcs=args.n_pcs,
            threads=args.threads,
        )
    elif args.command == "add-labels":
        an.add_labels(in_file=args.feature_file, label_file=args.label_file)
    elif args.command == "split-train-test":
        an.split_train_test(in_file=args.input, test_size=args.test_size)
    elif args.command == "train-model":
        gl = GapLearn()
        task = args.task
        conditional = args.conditional.lower() == "true"
        train_file = args.input.replace(".txt", f"_labeled_{task}_train.txt")
        metrics_file = args.metrics_file.replace(".txt", f"_{task}.txt")
        gl.train_cross_val(
            config_file=args.config_file,
            train_file=train_file,
            metrics_file=metrics_file,
            n_features=args.n_features,
        )

        if task == "Population" and conditional:
            labels_file = f"{os.path.dirname(args.input)}/labels_Superpopulation.txt"
            pops = pd.read_table(labels_file, header=0, sep="\t")["class_name"].unique().tolist()
            for pop in pops:
                try:
                    print(
                        f"Training conditional model for Population within Superpopulation: {pop}"
                    )
                    train_file = args.input.replace(
                        ".txt", f"_labeled_Population_within_{pop}_train.txt"
                    )
                    metrics_file = args.metrics_file.replace(
                        ".txt", f"_Population_within_{pop}.txt"
                    )
                    gl.train_cross_val(
                        config_file=args.config_file,
                        train_file=train_file,
                        metrics_file=metrics_file,
                        n_features=args.n_features,
                    )
                except Exception as e:
                    print(f"Error training model for Population within {pop}: {e}")

    elif args.command == "eval-model":
        gl = GapLearn()
        config_file = args.config_file
        task = args.task
        conditional = args.conditional.lower() == "true"
        train_file = args.input.replace(".txt", f"_labeled_{task}_train.txt")
        test_file = args.input.replace(".txt", f"_labeled_{task}_test.txt")
        metrics_file = args.metrics_file.replace(".txt", f"_{task}.txt")
        metrics_file_sorted = metrics_file.replace(".txt", "_sorted_best.txt")
        model_file = args.model_file.replace(".pkl", f"_{task}.pkl")

        gl.get_best_model_and_params(metrics_file=metrics_file)
        gl.final_fit_eval_on_full_train_then_eval_on_test(
            config_file=config_file,
            metrics_file=metrics_file_sorted,
            train_file=train_file,
            test_file=test_file,
            model_file=model_file,
        )

        if task == "Population" and conditional:
            labels_file = f"{os.path.dirname(args.input)}/labels_Superpopulation.txt"
            pops = pd.read_table(labels_file, header=0, sep="\t")["class_name"].unique().tolist()
            for pop in pops:
                try:
                    print(
                        "Evaluating conditional model for Population within "
                        f"Superpopulation: {pop}"
                    )
                    train_file = args.input.replace(
                        ".txt", f"_labeled_Population_within_{pop}_train.txt"
                    )
                    test_file = args.input.replace(
                        ".txt", f"_labeled_Population_within_{pop}_test.txt"
                    )
                    metrics_file = args.metrics_file.replace(
                        ".txt", f"_Population_within_{pop}.txt"
                    )
                    metrics_file_sorted = metrics_file.replace(".txt", "_sorted_best.txt")
                    model_file = args.model_file.replace(".pkl", f"_Population_within_{pop}.pkl")
                    gl.get_best_model_and_params(metrics_file=metrics_file)
                    gl.final_fit_eval_on_full_train_then_eval_on_test(
                        config_file=config_file,
                        metrics_file=metrics_file_sorted,
                        train_file=train_file,
                        test_file=test_file,
                        model_file=model_file,
                    )
                except Exception as e:
                    print(f"Error evaluating model for Population within {pop}: {e}")

    elif args.command == "predict":
        in_file = args.input
        task = args.task
        conditional = args.conditional.lower() == "true"
        metrics_file = args.metrics_file.replace(".txt", f"_{task}.txt")
        metrics_file_sorted = metrics_file.replace(".txt", "_sorted_best.txt")
        out_file = args.output.replace(".txt", f"_{task}.txt")
        label_file = args.label_file.replace(".txt", f"_{task}.txt")
        model_name = args.model_name
        model_file = args.model_file.replace(".pkl", f"_{task}.pkl")
        if not model_name:
            try:
                model_name = pd.read_table(
                    metrics_file_sorted, header=0, sep="\t"
                )["model"].iloc[0]
            except Exception as e:
                print(f"Cannot get the best model from metrics file {e}")
        print("Using model:", model_name)

        gl = GapLearn()
        gl.predict(
            in_file=in_file,
            out_file=out_file,
            label_file=label_file,
            model_name=model_name,
            model_file=model_file,
        )

        if task == "Population" and conditional:
            f = args.label_file.replace(".txt", "_Superpopulation.txt")
            pops = pd.read_table(f, header=0, sep="\t")["class_name"].unique().tolist()
            for pop in pops:
                try:
                    print(f"Predicting Population within Superpopulation: {pop}")
                    out_file = args.output.replace(".txt", f"_Population_within_{pop}.txt")
                    label_file = args.label_file.replace(".txt", f"_Population_within_{pop}.txt")
                    metrics_file = args.metrics_file.replace(
                        ".txt", f"_Population_within_{pop}.txt"
                    )
                    metrics_file_sorted = metrics_file.replace(".txt", "_sorted_best.txt")
                    model_file = args.model_file.replace(".pkl", f"_Population_within_{pop}.pkl")
                    model_name = pd.read_table(
                        metrics_file_sorted, header=0, sep="\t"
                    )["model"].iloc[0]
                    print("Using model:", model_name)
                    gl.predict(
                        in_file=in_file,
                        out_file=out_file,
                        label_file=label_file,
                        model_name=model_name,
                        model_file=model_file,
                    )
                except Exception as e:
                    print(f"Error predicting Population within {pop}: {e}")
    elif args.command == "summarize":
        conditional = args.conditional.lower() == "true"
        out_file = args.outfile
        an.get_summary_table(
            in_file_sp=args.input_sp,
            in_file_p=args.input_p,
            out_file=out_file,
            conditional=conditional,
        )

    elif args.command == "format-table":
        an.format_table(in_file=args.input, labels_dir=args.labels_dir, fmt=args.fmt)


if __name__ == "__main__":
    main()
