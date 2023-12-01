# This script is adapted from the first two scripts in
# https://github.com/jacobbien/trac-reproducible/tree/main/AmericanGut
# i.e., 0create_phyloseq_object.R and 1prep_data_all_levels.R

# Use the following command to install 'phyloseq'
# BiocManager::install('phyloseq')

library(phyloseq)

# import biom file (can take a few minutes due to large file size):
ag <- import_biom("original/8237_analysis.biom") 
# import metadata from mapping file:
map <- read.delim("original/8237_analysis_mapping.txt", 
                  sep = "\t",
                  header = TRUE, 
                  row.names = 1)
# assign metadata to phyloseq object:
sample_data(ag) <- map

# All fecal data
ag.fe <- subset_samples(ag, body_habitat == "UBERON:feces") ## only fecal samples

## Prune samples
depths <- colSums(ag.fe@otu_table@.Data) ## calculate sequencing depths

## Pruning (Minimum sequencing depth: at least 10000 reads per sample)
ag.filt1 <- prune_samples(depths > 10000, ag.fe)

## Extract taxa names and modify labels
tax <- ag.filt1@tax_table@.Data
# add an OTU column
tax <- cbind(tax, rownames(tax))
colnames(tax)[ncol(tax)] <- "OTU"
# make it so labels are unique
for (i in seq(2, 8)) {
  # add a number when the type is unknown... e.g. "g__"
  ii <- nchar(tax[, i]) == 3
  if (sum(ii) > 0)
    tax[ii, i] <- paste0(tax[ii, i], 1:sum(ii))
}
# cumulative labels are harder to read but easier to work with:
for (i in 2:8) {
  tax[, i] <- paste(tax[, i-1], tax[, i], sep = ";")
}
tax <- as.data.frame(tax, stringsAsFactors = TRUE)

## Aggregate data (only use species level data)
all_X <- as.data.frame(ag.filt1@otu_table@.Data)
all_X <- rowsum(all_X, tax$Rank7, reorder = TRUE)
rownames(all_X) <- sort(unique(tax$Rank7))
X_data <- t(all_X)
meta_data <- sample_data(ag.filt1)

## Save as csv files
write.csv(X_data, "X_data.csv", row.names = TRUE)
write.csv(meta_data, "meta_data.csv", row.names = TRUE)
