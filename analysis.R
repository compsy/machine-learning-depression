a <- read.csv('exports/merged_dataframe.csv', header=TRUE, sep=',')
print(names(a))

x <-cor(a, use='complete.obs')
#print(x)



# Correlations with pvalue
library(Hmisc)
x <- rcorr(as.matrix(a), type='pearson')


# Partial correlations
library(ggm)
pcors <- pcor(c("a4dkl.somScore", "cids.followup.somScore", 'aids.somScore'), var(a))
print(pcors)
