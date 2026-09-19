"""Miscellaneous helpers for merging, annotating, and benchmarking GAP tables."""

import os
from importlib import resources
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pylab as plt
from pySankey.sankey import sankey

BASE = resources.files(__package__.split(".")[0])
config_dir = f"{BASE}/config"


class Utils:
    """Helper routines for combining and annotating GAP ancestry tables."""

    def merge_tables(
        self,
        in_files: List[str] = [
            "OMNI_batch18/GenetcicAncestry_conditional.txt",
            "GDA_batch18/GenetcicAncestry_conditional.txt",
        ],
        out_file: str = "GAP_OMNI-GDA_conditional_batch18.txt",
    ) -> None:
        """Concatenate several tab-delimited tables into one output file.

        Args:
            in_files: Paths of the tab-delimited tables to concatenate.
            out_file: Path to write the concatenated table to.

        Raises:
            ValueError: If any file in ``in_files`` does not exist.
        """
        L = []
        for f in in_files:
            if not os.path.exists(f):
                raise ValueError(f"{f} does not exist. Please check the path.")
            else:
                df = pd.read_table(f, header=0, sep="\t")
                L.append(df)
        df_merged = pd.concat(L)
        df_merged.to_csv(out_file, header=True, index=False, sep="\t")

    def add_extra_info(
        self,
        in_file: str = "GAP_OMNI-GDA_batch18.txt",
        fam_files: List[str] = [
            "OMNI_batch18/OMNI_batch18_raw.fam",
            "GDA_batch18/GDA_batch18_raw.fam",
        ],
        qc_file: Optional[str] = None,
        extra_files: List[str] = [],
    ) -> None:
        """Annotate a GAP table with QC status, sample metadata, and project info.

        Args:
            in_file: Path to the input tab-delimited GAP table.
            fam_files: PLINK ``.fam`` files used to resolve sample names.
            qc_file: Optional file of sample IDs that passed QC, one per line.
            extra_files: Files providing array, batch, sex, and source metadata.

        Raises:
            ValueError: If ``qc_file`` or any file in ``extra_files`` does not
                exist.
        """
        df = pd.read_table(in_file, header=0, sep="\t")
        df.insert(1, "Extra", ".")
        df.insert(1, "QC", ".")
        df.insert(1, "RRID", ".")
        df.insert(1, "Project", ".")
        df.insert(1, "Batch", ".")
        df.insert(1, "Sex", ".")
        df.insert(1, "Source", ".")
        df.insert(1, "Array", ".")
        df.insert(1, "SampleName", ".")

        # adding QC column to indicate whether the sample passed QC
        if qc_file is not None:
            if not os.path.exists(qc_file):
                raise ValueError(f"{qc_file} does not exist. Please check the path.")
            df_qc = pd.read_table(qc_file, header=None, sep="\t")
            qc_samples = set(df_qc.iloc[:, 0].tolist())
            df["QC"] = ["PASS" if s in qc_samples else "FAIL" for s in df["SampleID"]]

        # adding Sex, Array, Source, Batch
        D = {}
        for ef in extra_files:
            if not os.path.exists(ef):
                raise ValueError(f"{ef} does not exist. Please check the path.")
            source = "Oxford" if ef.find("Oxford") != -1 else "Stanford"
            df_extra = pd.read_table(ef, header=None, sep="\t")
            for n in range(df_extra.shape[0]):
                fid = df_extra.iloc[n, 0]
                iid = df_extra.iloc[n, 1]
                array = df_extra.iloc[n, 2]
                batch = df_extra.iloc[n, 3]
                sex = df_extra.iloc[n, 4]
                k = f"{fid}_{iid}"
                D[k] = [array, batch, sex, source]

        v = [".", ".", ".", "."]
        df["Array"] = [D.get(s, v)[0] for s in df["SampleID"]]
        df["Batch"] = [D.get(s, v)[1] for s in df["SampleID"]]
        df["Sex"] = [D.get(s, v)[2] for s in df["SampleID"]]
        df["Source"] = [D.get(s, v)[3] for s in df["SampleID"]]

        # adding SampleName
        D = {}
        for fam_file in fam_files:
            df_fam = pd.read_table(fam_file, header=None, sep=" ")
            for n in range(df_fam.shape[0]):
                iid = df_fam.iloc[n, 1]
                fid = df_fam.iloc[n, 0]
                sample_id = f"{fid}_{iid}"
                sample_name = iid
                D[sample_id] = sample_name
        df["SampleName"] = [
            self._update_sample_name(D.get(s, s)) for s in df["SampleID"]
        ]

        # adding Project
        df["Project"] = [self._get_project_name(s) for s in df["SampleName"]]

        out_file = in_file.replace(".txt", "_extraInfo.txt")
        df.to_csv(out_file, header=True, index=False, sep="\t")

    def _update_sample_name(self, sample: str) -> str:
        """Normalize a raw sample name to its canonical form.

        Args:
            sample: Raw sample name as read from a fam/extra-info file.

        Returns:
            The normalized sample name.
        """
        sample0 = sample
        sample = sample.replace("DNA", "")
        if sample.find("EDMN") != -1:
            if sample.find("_") != -1:
                sample = sample.split("_")[-1].replace("Bh", "R")
                if sample.find("R") == -1:
                    sample = "R" + sample
            else:
                sample = (
                    sample.replace("EDMN", "R")
                    .replace("FFPE", "")
                    .replace("ISLET", "")
                    .replace("REEXTRACTED", "")
                )

        elif sample.find("HPAP") != -1:
            if sample.find("_") != -1:
                sample = "HPAP" + sample.split("_")[-1]

        elif sample.find("HIPP") != -1:
            sample = sample.replace("FFPE_", "").replace("FFPE", "")
        elif sample.find("HIG") != -1:
            sample = sample.replace("HIG1", "HIGI")
            if sample.find("HIGI") == -1:
                sample = sample.replace("HIG", "HIGI")

        elif sample.find("ST1DR") != -1:
            sample = sample.split(".")[0]
        elif sample.find("SDRC") != -1:
            sample = sample.replace("SDRC", "ST1DR")
        elif sample.find("DIMC") != -1:
            sample = "ST1DR" + sample.split("_")[-1]

        elif sample.find("IPOP") != -1:
            if sample.find("_") != -1:
                sample = "IPOP" + sample.split("_")[-1]

        return sample

    def _get_project_name(self, sample: str) -> str:
        """Infer the source project of a sample from its normalized name.

        Args:
            sample: Normalized sample name, as returned by
                :meth:`_update_sample_name`.

        Returns:
            The project name, or ``"."`` if it cannot be determined.
        """
        project = "."
        if sample.find("IIDP") != -1:
            project = "IIDP"
        elif sample.find("HPAP") != -1:
            project = "HPAP"
        elif sample.find("HIGI") != -1:
            project = "IIDP"
        elif sample.find("HIPP") != -1:
            project = "IIDP"
        elif sample.find("ST1DR") != -1:
            project = "ST1DR"
        elif sample.find("R") == 0:
            project = "ADI"
        return project

    def benchmark_against_self_reported_race(
        self,
        in_file: str,
        in_file2: str,
        cols: List[str] = ["SampleID", "ID", "Superpopulation", "Race"],
    ) -> None:
        """Plot a Sankey diagram comparing predicted ancestry to self-reported race.

        Args:
            in_file: Path to the table containing predicted ancestry.
            in_file2: Path to the table containing self-reported race.
            cols: Column names ``[id_col1, id_col2, ancestry_col, race_col]``
                used to join the two tables and select the columns to plot.
        """
        df = pd.read_table(in_file, header=0, sep="\t")
        df2 = pd.read_table(in_file2, header=0, sep="\t")
        df_merged = pd.merge(df, df2, left_on=cols[0], right_on=cols[1])
        sankey(df_merged[cols[2]], df_merged[cols[3]], aspect=30, fontsize=10)
        plt.tight_layout()
        out_file = in_file.split(".txt")[0] + "_vs_" + in_file2.split(".txt")[0] + "_sankey.png"
        plt.savefig(out_file)
        print(f"{df_merged.shape[0]} samples shared")
        print(f"sankey plot saved to {out_file}")
