We present our implementation of a cross-experiment prediction model for the **Genomic to Artificial Sequences (G2A) primary track of the 2024 IBIS challenge**. https://ibis.autosome.org/home

### Download datasets
https://ibis.autosome.org/download_data/final

### Description
Both ChIP-seq and GHTS data were used for training. First, 40-nt-long DNA sequences that
flanked the abs_summit coordinates were retrieved for all transcription factors (TFs). Then,
the sequences were split into overlapping 8-mers with a 1-nt step. To achieve equivariance,
reverse complement sequences were generated for each 8-mer, and then each pair of k-
mers was sorted in alphabetical order. In each pair, only the first k-mer was retained to
represent the whole sense-antisense DNA pair. The resulting list of tokens for each label was
processed with a TfIdfVectorizer to obtain embeddings that reflect the frequency of k-mer (or
a group of k-mers) occurrence in the DNA sequence flanking the abs_summit of the
transcription factor (TF) of interest, as opposed to all TFs in the dataset. TF-IDF embeddings
were created for a range of n-grams from 1 to 3 in order to account for longer TF binding
sites. The data were fitted into a StackingClassifier that comprised LogisticRegression and
KNN as initial estimators and CatBoost, a gradient boosting classifier, as the final one. All
embeddings except those of the target class were used as negatives during model training.
