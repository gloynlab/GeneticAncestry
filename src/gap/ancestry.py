import os
import pandas as pd
import subprocess
from .models import GapLearn
from sklearn.model_selection import train_test_split
from .utils import *

class Ancestry:
    def __init__(self):
        self.url_1000genomes_vcf = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/1000G_2504_high_coverage/' \
                + 'working/20201028_3202_phased/CCDG_14151_B01_GRM_WGS_2020-08-05_{}.filtered.shapeit2-duohmm-phased.vcf.gz'

        self.url_1000genomes_sample_populatioin = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/' \
                + '1000G_2504_high_coverage/20130606_g1k_3202_samples_ped_population.txt'

        self.url_1000genomes_sample_unrelated = 'http://ftp.1000genomes.ebi.ac.uk/vol1/ftp/data_collections/' \
                + '1000G_2504_high_coverage/1000G_2504_high_coverage.sequence.index'

        self.chs = [f'chr{i}' for i in range(1, 23)]

    def get_reference_data(self, out_dir='ref', source='1000genomes', url=None, unrelated_samples_only=True):
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)

        if source == '1000genomes':
            if not url:
                self.url = self.url_1000genomes_vcf
                vcfs = []
                for ch in self.chs:
                    in_file = self.url.format(ch)
                    in_file_idx = in_file + '.tbi'
                    out_file = os.path.join(out_dir, f'{source}_{ch}.vcf.gz')
                    out_file_idx = os.path.join(out_dir, f'{source}_{ch}.vcf.gz.tbi')
                    if os.path.exists(out_file) and os.path.exists(out_file_idx):
                        print(f"{out_file} and {out_file_idx} already exist. Skipping download.")
                        vcfs.append(out_file)
                    else:
                        try:
                            cmd = f'wget {in_file} -O {out_file}'
                            cmd_idx = f'wget {in_file_idx} -O {out_file_idx}'
                            print(cmd, flush=True)
                            subprocess.run(cmd, shell=True)
                            print(cmd_idx, flush=True)
                            subprocess.run(cmd_idx, shell=True)
                            vcfs.append(out_file)
                        except Exception as e:
                            print(f"Error downloading {in_file}: {e}")

            for ch in self.chs:
                in_file = os.path.join(out_dir, f'{source}_{ch}.vcf.gz')
                in_file_idx = in_file + '.tbi'
                if not os.path.exists(in_file) or not os.path.exists(in_file_idx):
                    raise ValueError(f"{in_file} or {in_file_idx} not found.")
                else:
                    for f in [in_file, in_file_idx]:
                        cmd = f'gzip -t {f}'
                        result = subprocess.run(cmd, capture_output=True, text=True, shell=True)
                        if result.returncode != 0:
                            raise ValueError(f"{f} is not a valid gzip file. Please remove the file and try downloading again.")
                        else:
                            print(f"{f} is valid", flush=True)

            out_vcf = os.path.join(out_dir, f'{source}.vcf.gz')
            self._concat_vcfs(vcfs, out_vcf)

            if unrelated_samples_only:
                sampleInfo_file = os.path.join(out_dir, f'{source}_unrelated_sampleInfo.txt')
                sampleID_file = sampleInfo_file.replace('.txt', '_sampleID.txt')
                self._get_unrelated_samples(sampleInfo_file, sampleID_file)
                extracted_file = os.path.join(out_dir, f'{source}_unrelated.vcf.gz')
                self._extract_samples(out_vcf, sampleID_file, extracted_file)

    def merge_dataset_with_reference(self, dataset_vcf, reference_vcf, out_file, r2_threshold=None, threads=4):
        try:
            out_dir = os.path.dirname(out_file)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            if r2_threshold is not None:
                dataset_vcf_in = dataset_vcf
                dataset_vcf = dataset_vcf.split('.vcf')[0] + f'_R2_{r2_threshold}.vcf.gz'
                cmd = f'bcftools filter --threads {threads} -i "R2 > {r2_threshold}" {dataset_vcf_in} -Oz -o {dataset_vcf}'
                subprocess.run(cmd, shell=True, check=True)

            dataset_bed = dataset_vcf.split('.vcf')[0]
            reference_bed = reference_vcf.split('.vcf')[0]

            try:
                cmd = f'plink --keep-allele-order --double-id --vcf {dataset_vcf} --make-bed --out {dataset_bed} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
            except:
                raise ValueError(f"Error converting {dataset_vcf} to bed/bim/fam. Please check the VCF file.")

            if os.path.exists(reference_bed + '.bed') and os.path.exists(reference_bed + '.bim') and os.path.exists(reference_bed + '.fam'):
                print(f"{reference_bed} bed/bim/fam already exist. Skipping conversion from vcf.")
            else:
                cmd = f'plink --keep-allele-order --vcf {reference_vcf} --make-bed --out {reference_bed} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)

            if not os.path.exists(dataset_bed + '.bim') or not os.path.exists(reference_bed + '.bim'):
                print("bed/bim/fam files not found for dataset or reference. Cannot proceed with merging.")
                return

            shared_variants_file = dataset_bed + '_shared_variants.txt'
            self._get_shared_variants(dataset_bed + '.bim', reference_bed + '.bim', shared_variants_file, update_bim=True)

            dataset_extracted_bed = dataset_bed + '_extracted'
            reference_extracted_bed = reference_bed + '_extracted'

            cmd = f'plink --keep-allele-order --bfile {dataset_bed} --extract {shared_variants_file} --make-bed --out {dataset_extracted_bed}'
            subprocess.run(cmd, shell=True, check=True)
            cmd = f'plink --keep-allele-order --bfile {reference_bed} --extract {shared_variants_file} --make-bed --out {reference_extracted_bed}'
            subprocess.run(cmd, shell=True, check=True)

            cmd = f'plink --keep-allele-order --bfile {reference_extracted_bed} --bmerge {dataset_extracted_bed} --make-bed --out {out_file} --threads {threads}'
            subprocess.run(cmd, shell=True, check=True)

            os.remove(dataset_extracted_bed + '.bed')
            os.remove(dataset_extracted_bed + '.bim')
            os.remove(dataset_extracted_bed + '.fam')
            os.remove(reference_extracted_bed + '.bed')
            os.remove(reference_extracted_bed + '.bim')
            os.remove(reference_extracted_bed + '.fam')
            os.remove(shared_variants_file)
        except Exception as e:
            print(f"Error merging VCFs: {e}")

    def _get_shared_variants(self, dataset_bim, reference_bim, out_file, update_bim=True):
        try:
            df1 = pd.read_table(dataset_bim, header=None, sep='\t')
            df1.iloc[:, 1] = df1.apply(lambda row: ':'.join([str(row[0]), str(row[3]), str(row[5]), str(row[4])]), axis=1)
            df2 = pd.read_table(reference_bim, header=None, sep='\t')
            df2.iloc[:, 1] = df2.apply(lambda row: ':'.join([str(row[0]), str(row[3]), str(row[5]), str(row[4])]), axis=1)
            shared_variants = set(df1.iloc[:, 1]).intersection(set(df2.iloc[:, 1])) 
            with open(out_file, 'w') as f:
                for n in range(df2.shape[0]):
                    k = df2.iloc[n, 1]
                    if k in shared_variants:
                        f.write(k + '\n')
            print(f"Shared variants saved to {out_file}")
            if update_bim:
                df1.to_csv(dataset_bim, header=False, index=False, sep='\t')
                df2.to_csv(reference_bim, header=False, index=False, sep='\t')
        except Exception as e:
            print(f"Error getting shared variants: {e}")

    def feature_engineering(self, in_file, out_file='data/features.txt', pruning=True, pruning_params=[50, 5, 0.2], pca=True, n_pcs=50, threads=4):
        out_file_bed = f'{in_file}_pruned'
        out_file_pca = f'{in_file}_pca'
        infile = in_file
        try:
            if pruning:
                cmd = f'plink --keep-allele-order --bfile {in_file} --indep-pairwise {pruning_params[0]} {pruning_params[1]} {pruning_params[2]} --out {in_file} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
                cmd = f'plink --keep-allele-order --bfile {in_file} --extract {in_file}.prune.in --make-bed --out {out_file_bed} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
                in_file = out_file_bed

            if pca:
                cmd = f'plink --bfile {in_file} --pca {n_pcs} --out {out_file_pca} --threads {threads}'
                subprocess.run(cmd, shell=True, check=True)
                print(f"PCA results saved to {out_file_pca}.eigenvec")

                df = pd.read_table(f'{out_file_pca}.eigenvec', header=None, sep=' ')
                df.columns = ['FID', 'IID'] + [f'PC{i}' for i in range(1, df.shape[1]-1)]
                df.to_csv(out_file, header=True, index=False, sep='\t')
                print(f"Features saved to {out_file}")
            else:
                print('PCA is the only feature engineering method implemented currently.')
                return
        except Exception as e:
            print(f"Error in feature engineering: {e}")

    def add_labels(self, in_file='data/features.txt', label_file='ref/1000genomes_unrelated_sampleInfo.txt'):
        D = {}
        df_labels = pd.read_table(label_file, header=0, sep='\t')
        for n in range(df_labels.shape[0]):
            k = df_labels['SampleID'].iloc[n]
            D[k] = [df_labels['Population'].iloc[n], df_labels['Superpopulation'].iloc[n]]

        df = pd.read_table(in_file, header=0, sep='\t')
        df.columns = ['FID', 'SampleID'] + list(df.columns)[2:]
        dfp = df.iloc[:, 1:].copy()
        dfsp = df.iloc[:, 1:].copy()
        dfp['class_name'] = [D[k][0] if k in D else '.' for k in dfp['SampleID']]
        dfsp['class_name'] = [D[k][1] if k in D else '.' for k in dfsp['SampleID']]

        dfp_labeled = dfp[dfp['class_name'] != '.'].copy()
        dfsp_labeled = dfsp[dfsp['class_name'] != '.'].copy()
        dfsp_unlabeled = dfsp[dfsp['class_name'] == '.'].copy()
        dfsp_unlabeled.drop(columns=['class_name'], inplace=True)

        labels_file_p = f'{os.path.dirname(in_file)}/labels_Population.txt'
        labels_file_sp = f'{os.path.dirname(in_file)}/labels_Superpopulation.txt'
        p_labels = sorted(dfp_labeled['class_name'].unique())
        sp_labels = sorted(dfsp_labeled['class_name'].unique())
        df_p_labels = pd.DataFrame({'class_name': p_labels, 'class': range(len(p_labels))})
        df_sp_labels = pd.DataFrame({'class_name': sp_labels, 'class': range(len(sp_labels))})
        df_p_labels.to_csv(labels_file_p, header=True, index=False, sep='\t')
        df_sp_labels.to_csv(labels_file_sp, header=True, index=False, sep='\t')
        print(f"Superpopulation labels saved to {labels_file_sp}")
        print(f"Population labels saved to {labels_file_p}")

        dfp_labeled['class'] = [p_labels.index(x) for x in dfp_labeled['class_name']]
        dfsp_labeled['class'] = [sp_labels.index(x) for x in dfsp_labeled['class_name']]

        labeled_file_p = in_file.replace('.txt', '_labeled_Population.txt')
        labeled_file_sp = in_file.replace('.txt', '_labeled_Superpopulation.txt')
        to_predict_file = f'{os.path.dirname(in_file)}/ToPredict.txt'
        dfp_labeled.to_csv(labeled_file_p, header=True, index=False, sep='\t')
        dfsp_labeled.to_csv(labeled_file_sp, header=True, index=False, sep='\t')
        dfsp_unlabeled.to_csv(to_predict_file, header=True, index=False, sep='\t')
        print(f"Labeled Superpopulation data saved to {labeled_file_sp}")
        print(f"Labeled Population data saved to {labeled_file_p}")
        print(f"To predict data saved to {to_predict_file}")

    def split_train_test(self, in_file, test_size=0.2, conditional=True, random_state=42):
        for pop in ['Superpopulation', 'Population']:
            in_f = in_file.replace(f'.txt', f'_labeled_{pop}.txt')
            train_f = in_f.replace('.txt', '_train.txt')
            test_f = in_f.replace('.txt', '_test.txt')
            df = pd.read_table(in_f, header=0, sep='\t')
            df_train, df_test = train_test_split(df, test_size=test_size, random_state=random_state, stratify=df['class'])
            df_train.to_csv(train_f, header=True, index=False, sep='\t')
            df_test.to_csv(test_f, header=True, index=False, sep='\t')
            print(f"{pop} train data saved to {train_f}")
            print(f"{pop} test data saved to {test_f}")
        if conditional:
            D = {}
            labeled_file_sp = in_file.replace('.txt', '_labeled_Superpopulation.txt')
            labeled_file_p = in_file.replace('.txt', '_labeled_Population.txt')

            df_sp = pd.read_table(labeled_file_sp, header=0, sep='\t')
            df_p = pd.read_table(labeled_file_p, header=0, sep='\t')
            for n in range(df_sp.shape[0]):
                k = df_sp['SampleID'].iloc[n]
                class_name = df_sp['class_name'].iloc[n]
                D.setdefault(class_name, [])
                D[class_name].append(k)

            for pop in sorted(D.keys()):
                sample_ids = D[pop]
                df_sub = df_p[df_p['SampleID'].isin(sample_ids)].copy()

                p_labels = sorted(df_sub['class_name'].unique())
                df_sub['class'] = [p_labels.index(x) for x in df_sub['class_name']]
                df_p_labels = pd.DataFrame({'class_name': p_labels, 'class': range(len(p_labels))})
                out_file_lables = f'{os.path.dirname(in_file)}/labels_Population_within_{pop}.txt'
                df_p_labels.to_csv(out_file_lables, header=True, index=False, sep='\t')
                print(f"Population labels within {pop} saved to {out_file_lables}")

                sub_f = in_file.replace('.txt', f'_labeled_Population_within_{pop}.txt')
                train_f = in_file.replace('.txt', f'_labeled_Population_within_{pop}_train.txt')
                test_f = in_file.replace('.txt', f'_labeled_Population_within_{pop}_test.txt')
                df_train, df_test = train_test_split(df_sub, test_size=test_size, random_state=random_state, stratify=df_sub['class'])
                df_sub.to_csv(sub_f, header=True, index=False, sep='\t')
                df_train.to_csv(train_f, header=True, index=False, sep='\t')
                df_test.to_csv(test_f, header=True, index=False, sep='\t')
                print(f"Population data within {pop} saved to {sub_f}")
                print(f"Population train data within {pop} saved to {train_f}")
                print(f"Population test data within {pop} saved to {test_f}")

    def _concat_vcfs(self, vcf_list, out_vcf, remove_downloaded=True):
        try:
            print(f'Concatenating VCFs into {out_vcf}', flush=True)
            cmd = ['bcftools', 'concat', '-Oz', '-o', out_vcf] + vcf_list
            res1 = subprocess.run(cmd, check=True, capture_output=True)
            cmd = ['bcftools', 'index', out_vcf]
            res2 = subprocess.run(cmd, check=True, capture_output=True)
            if res1.returncode == 0 and res2.returncode == 0:
                print(f"Concatenated VCF saved to {out_vcf}")
                if remove_downloaded:
                    for f in vcf_list:
                        f2 = f + '.tbi'
                        if os.path.exists(f):
                            os.remove(f)
                        if os.path.exists(f2):
                            os.remove(f2)
            else:
                raise ValueError(f"Error concatenating VCFs")
        except Exception as e:
            print(f"Error concatenating VCFs: {e}")

    def _get_unrelated_samples(self, out_file, out_file2):
        in_file1 = self.url_1000genomes_sample_populatioin
        in_file2 = self.url_1000genomes_sample_unrelated

        df1 = pd.read_table(in_file1, header=0, sep=' ')
        df2 = pd.read_table(in_file2, comment='#', header=None)

        df1_sub = df1.loc[df1['SampleID'].isin(df2[9].str.strip())]
        df1_sub.to_csv(out_file, header=True, index=False, sep='\t')
        df1_sub['SampleID'].to_csv(out_file2, header=False, index=False, sep='\t')

    def _extract_samples(self, vcf_file, sample_file, out_file):
        try:
            print(f'Extracting unrelated samples')
            cmd = ['bcftools', 'view', '-S', sample_file, '-Oz', '-o', out_file, vcf_file]
            subprocess.run(cmd, check=True)
            subprocess.run(['bcftools', 'index', out_file], check=True)
            print(f"Extracted samples VCF saved to {out_file}")
        except Exception as e:
            print(f"Error extracting samples: {e}")

    def get_summary_table(self, in_file_sp='data/Predicted_Superpopulation.txt', in_file_p='data/Predicted_Population.txt', out_file='GeneticAncestry.txt', conditional=True):
        df_sp = pd.read_table(in_file_sp, header=0, sep='\t')
        df_p = pd.read_table(in_file_p, header=0, sep='\t')
        df_merged = pd.merge(df_sp, df_p, on='SampleID', suffixes=('_Superpopulation', '_Population'))
        df = pd.DataFrame()
        df['SampleID'] = df_merged['SampleID']
        df['Superpopulation'] = df_merged['class_Superpopulation']
        df['Population'] = df_merged['class_Population']
        for col in df_merged.columns:
            if col not in ['SampleID', 'class_Superpopulation', 'class_Population']:
                df[col] = df_merged[col] 
        df.to_csv(out_file, header=True, index=False, sep='\t')
        print(f"Genetic ancestry table saved to {out_file}")
        if conditional:
            out_file_cond = out_file.replace('.txt', '_conditional.txt')
            df_sp = pd.read_table(in_file_sp, header=0, sep='\t')
            labels_sp = pd.read_table(in_file_sp.replace('Predicted', 'labels'), header=0, sep='\t')['class_name'].tolist()
            L = [df_sp]
            for pop in labels_sp:
                in_file = in_file_p.replace('.txt', f'_within_{pop}.txt')
                df_p = pd.read_table(in_file, header=0, sep='\t')
                df_p.columns = [f'{col}_within_{pop}' for col in df_p.columns]
                L.append(df_p)
            df_merged = pd.concat(L, axis=1)

            df = pd.DataFrame()
            df['SampleID'] = df_merged['SampleID']
            df['Superpopulation'] = df_merged['class']

            L = []
            for n in range(df_merged.shape[0]):
                sp = df_merged['class'].iloc[n]
                p = df_merged[f'class_within_{sp}'].iloc[n] 
                L.append(p)
            df['Population'] = L

            for pop in labels_sp:
                df[pop] = df_merged[pop]

            pops = {}
            for p in df_merged.columns:
                if p.find('_within_') != -1:
                    pop = p.split('_within_')[0]
                    if pop not in ['SampleID', 'class']:
                        pops[pop] = df_merged[p] 
            for pop in sorted(pops):
                df[pop] = pops[pop]

            df.to_csv(out_file_cond, header=True, index=False, sep='\t')
            print(f"Genetic ancestry (conditional) table saved to {out_file_cond}")

    def format_table(self, in_file='GenetcicAncestry.txt', labels_dir='data', fmt='IIDP'):
        D = {}
        for f in os.listdir(labels_dir):
            if f.startswith('labels_Population_within_') and f.endswith('.txt'):
                df = pd.read_table(os.path.join(labels_dir, f), header=0, sep='\t')
                for n in range(df.shape[0]):
                    p = df.iloc[n, 0]
                    sp = f.split('labels_Population_within_')[1].split('.txt')[0]
                    D[p] = sp

        out_file = in_file.replace('.txt', f'_{fmt}.txt')
        df = pd.read_table(in_file, header=0, sep='\t')
        if fmt == 'IIDP':
            df_formatted = pd.DataFrame()
            df_formatted['Sample'] = df['SampleID']
            df_formatted['SuperpopulationClass'] = df['Superpopulation']
            df_formatted['PopulationClass'] = df['Population']

            L = []
            for n in range(df.shape[0]):
                sp = df['Superpopulation'].iloc[n]
                p = df['Population'].iloc[n]
                consistency = 'Consistent' if D.get(p) == sp else 'InConsistent'
                L.append(consistency)
            df_formatted['PredictionConsistency'] = L

            L = []
            for n in range(df.shape[0]):
                V = []
                for m in range(3, 8):
                    pop = df.columns[m]
                    V.append(f"{pop}:{df.iloc[n, m]:.4f}")
                L.append('|'.join(V))
            df_formatted['SuperPopulationPossibility'] = L

            L = []
            for n in range(df.shape[0]):
                V = []
                for m in range(8, df.shape[1]):
                    pop = df.columns[m]
                    V.append(f"{pop}:{df.iloc[n, m]:.4f}")
                L.append('|'.join(V))
            df_formatted['PopulationPossibility'] = L

            df_formatted.to_csv(out_file, header=True, index=False, sep='\t')

        else:
            print(f"Format {fmt} not recognized. No formatting applied.")
            return
